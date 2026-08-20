// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::Arc;

use arrow::datatypes::{Schema, SchemaRef};
use async_ffi::{FfiFuture, FutureExt};
use async_trait::async_trait;
use datafusion_catalog::Session;
use datafusion_common::error::{DataFusionError, Result};
use datafusion_common::{GetExt, Statistics, internal_datafusion_err, not_impl_err};
use datafusion_datasource::TableSchema;
use datafusion_datasource::file::FileSource;
use datafusion_datasource::file_compression_type::FileCompressionType;
use datafusion_datasource::file_format::{FileFormat, FileFormatFactory};
use datafusion_datasource::file_scan_config::FileScanConfig;
use datafusion_datasource::file_sink_config::FileSinkConfig;
use datafusion_execution::{FunctionRegistry, TaskContext};
use datafusion_expr::physical_planning_context::ScalarSubqueryResults;
use datafusion_expr::{AggregateUDF, ScalarUDF, WindowUDF};
use datafusion_physical_expr::{LexRequirement, PhysicalExpr, PhysicalSortExpr};
use datafusion_physical_expr_common::sort_expr::{LexOrdering, PhysicalSortRequirement};
use datafusion_physical_plan::ExecutionPlan;
use datafusion_physical_plan::proto::{
    ExecutionPlanDecode, ExecutionPlanDecodeCtx, ExecutionPlanEncode,
    ExecutionPlanEncodeCtx,
};
use datafusion_proto::logical_plan::{
    DefaultLogicalExtensionCodec, LogicalExtensionCodec,
};
use datafusion_proto::physical_plan::from_proto::parse_physical_expr;
use datafusion_proto::physical_plan::to_proto::serialize_physical_expr;
use datafusion_proto::physical_plan::{
    DefaultPhysicalExtensionCodec, PhysicalExtensionCodec,
};
use datafusion_proto::protobuf::{
    self, FileSinkConfig as ProtoFileSinkConfig, PhysicalExprNode, PhysicalPlanNode,
};
use object_store::{ObjectMeta, ObjectStore};
use prost::Message;

use stabby::string::String as SString;
use stabby::vec::Vec as SVec;
use tokio::runtime::Handle;

use crate::execution::FFI_TaskContextProvider;
use crate::execution_plan::FFI_ExecutionPlan;
use crate::physical_expr::sort::FFI_PhysicalSortExpr;
use crate::proto::logical_extension_codec::FFI_LogicalExtensionCodec;
use crate::proto::physical_extension_codec::FFI_PhysicalExtensionCodec;
use crate::session::{FFI_SessionRef, ForeignSession};
use crate::util::{FFI_Option, FFI_Result};
use crate::{df_result, sresult_return};

/// Adapter that backs [`ExecutionPlanEncodeCtx`] / [`ExecutionPlanDecodeCtx`]
/// for the [`FileScanConfig`] / [`FileSinkConfig`] (de)serialization performed
/// in this module, using a crossed [`FFI_PhysicalExtensionCodec`] instead of a
/// local `datafusion-proto` codec.
///
/// `FileScanConfig::try_to_proto`/`try_from_proto` only ever touch
/// `encode_expr`/`decode_expr` (for sort/partitioning/projection expressions)
/// and never a child execution plan, so `encode_plan`/`decode_plan` are wired
/// as `not_impl_err!` here rather than implemented against a full recursive
/// plan (de)serializer.
struct FfiPlanCodecAdapter<'a> {
    codec: &'a dyn PhysicalExtensionCodec,
    task_ctx: &'a TaskContext,
}

impl ExecutionPlanEncode for FfiPlanCodecAdapter<'_> {
    fn encode_plan(&self, _plan: &Arc<dyn ExecutionPlan>) -> Result<PhysicalPlanNode> {
        not_impl_err!(
            "FFI FileFormat does not support encoding child execution plans; \
             FileScanConfig/FileSinkConfig never carry one"
        )
    }

    fn encode_expr(&self, expr: &Arc<dyn PhysicalExpr>) -> Result<PhysicalExprNode> {
        serialize_physical_expr(expr, self.codec)
    }

    fn encode_udf(&self, udf: &ScalarUDF) -> Result<Option<Vec<u8>>> {
        let mut buf = vec![];
        self.codec.try_encode_udf(udf, &mut buf)?;
        Ok((!buf.is_empty()).then_some(buf))
    }

    fn encode_udaf(&self, udaf: &AggregateUDF) -> Result<Option<Vec<u8>>> {
        let mut buf = vec![];
        self.codec.try_encode_udaf(udaf, &mut buf)?;
        Ok((!buf.is_empty()).then_some(buf))
    }

    fn encode_udwf(&self, udwf: &WindowUDF) -> Result<Option<Vec<u8>>> {
        let mut buf = vec![];
        self.codec.try_encode_udwf(udwf, &mut buf)?;
        Ok((!buf.is_empty()).then_some(buf))
    }
}

impl ExecutionPlanDecode for FfiPlanCodecAdapter<'_> {
    fn decode_plan(&self, _node: &PhysicalPlanNode) -> Result<Arc<dyn ExecutionPlan>> {
        not_impl_err!(
            "FFI FileFormat does not support decoding child execution plans; \
             FileScanConfig/FileSinkConfig never carry one"
        )
    }

    fn decode_plan_with_scalar_subquery_results(
        &self,
        _node: &PhysicalPlanNode,
        _results: ScalarSubqueryResults,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        not_impl_err!(
            "FFI FileFormat does not support decoding child execution plans; \
             FileScanConfig/FileSinkConfig never carry one"
        )
    }

    fn decode_expr(
        &self,
        node: &PhysicalExprNode,
        input_schema: &Schema,
    ) -> Result<Arc<dyn PhysicalExpr>> {
        parse_physical_expr(node, self.task_ctx, input_schema, self.codec)
    }

    fn task_ctx(&self) -> &TaskContext {
        self.task_ctx
    }

    fn decode_udf(&self, name: &str, payload: Option<&[u8]>) -> Result<Arc<ScalarUDF>> {
        match payload {
            Some(buf) => self.codec.try_decode_udf(name, buf),
            None => self
                .task_ctx
                .udf(name)
                .or_else(|_| self.codec.try_decode_udf(name, &[])),
        }
    }

    fn decode_udaf(
        &self,
        name: &str,
        payload: Option<&[u8]>,
    ) -> Result<Arc<AggregateUDF>> {
        match payload {
            Some(buf) => self.codec.try_decode_udaf(name, buf),
            None => self
                .task_ctx
                .udaf(name)
                .or_else(|_| self.codec.try_decode_udaf(name, &[])),
        }
    }

    fn decode_udwf(&self, name: &str, payload: Option<&[u8]>) -> Result<Arc<WindowUDF>> {
        match payload {
            Some(buf) => self.codec.try_decode_udwf(name, buf),
            None => self
                .task_ctx
                .udwf(name)
                .or_else(|_| self.codec.try_decode_udwf(name, &[])),
        }
    }
}

/// FFI safe version of [`FileCompressionType`]'s inner
/// [`CompressionTypeVariant`](datafusion_common::parsers::CompressionTypeVariant).
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FFI_FileCompressionType {
    Gzip,
    Bzip2,
    Xz,
    Zstd,
    Uncompressed,
}

impl From<FileCompressionType> for FFI_FileCompressionType {
    fn from(value: FileCompressionType) -> Self {
        use datafusion_common::parsers::CompressionTypeVariant;
        match value.get_variant() {
            CompressionTypeVariant::GZIP => Self::Gzip,
            CompressionTypeVariant::BZIP2 => Self::Bzip2,
            CompressionTypeVariant::XZ => Self::Xz,
            CompressionTypeVariant::ZSTD => Self::Zstd,
            CompressionTypeVariant::UNCOMPRESSED => Self::Uncompressed,
        }
    }
}

impl From<FFI_FileCompressionType> for FileCompressionType {
    fn from(value: FFI_FileCompressionType) -> Self {
        match value {
            FFI_FileCompressionType::Gzip => FileCompressionType::GZIP,
            FFI_FileCompressionType::Bzip2 => FileCompressionType::BZIP2,
            FFI_FileCompressionType::Xz => FileCompressionType::XZ,
            FFI_FileCompressionType::Zstd => FileCompressionType::ZSTD,
            FFI_FileCompressionType::Uncompressed => FileCompressionType::UNCOMPRESSED,
        }
    }
}

/// A stable struct for sharing [`FileFormatFactory`] across FFI boundaries.
/// See `FFI_TableProvider` in [`crate::table_provider`] for a description of
/// the `Foreign`/`FFI_` naming convention used throughout this crate.
#[repr(C)]
#[derive(Debug)]
pub struct FFI_FileFormatFactory {
    /// Returns the default file extension for this format, e.g. "csv"
    get_ext: unsafe extern "C" fn(&Self) -> SString,

    /// Initialize a [`FileFormat`] configured from session and command level
    /// options.
    create: unsafe extern "C" fn(
        &Self,
        session: FFI_SessionRef,
        format_options: SVec<(SString, SString)>,
    ) -> FFI_Result<FFI_FileFormat>,

    /// Initialize a [`FileFormat`] with all options set to default values.
    default: unsafe extern "C" fn(&Self) -> FFI_FileFormat,

    /// Used to construct a [`FFI_SessionRef`] for the [`Self::create`] call.
    pub logical_codec: FFI_LogicalExtensionCodec,

    /// Used to create a clone on the provider of the factory. This should
    /// only need to be called by the receiver of the factory.
    clone: unsafe extern "C" fn(factory: &Self) -> Self,

    /// Release the memory of the private data when it is no longer being used.
    release: unsafe extern "C" fn(arg: &mut Self),

    /// Return the major DataFusion version number of this provider.
    pub version: unsafe extern "C" fn() -> u64,

    /// Internal data. This is only to be accessed by the provider of the
    /// factory. A [`ForeignFileFormatFactory`] should never attempt to
    /// access this data.
    private_data: *mut c_void,

    /// Utility to identify when FFI objects are accessed locally through
    /// the foreign interface. See [`crate::get_library_marker_id`] and
    /// the crate's `README.md` for more information.
    pub library_marker_id: extern "C" fn() -> usize,
}

unsafe impl Send for FFI_FileFormatFactory {}
unsafe impl Sync for FFI_FileFormatFactory {}

struct FormatFactoryPrivateData {
    factory: Arc<dyn FileFormatFactory>,
    runtime: Option<Handle>,
}

impl FFI_FileFormatFactory {
    fn inner(&self) -> &Arc<dyn FileFormatFactory> {
        let private_data = self.private_data as *const FormatFactoryPrivateData;
        unsafe { &(*private_data).factory }
    }

    fn runtime(&self) -> &Option<Handle> {
        let private_data = self.private_data as *const FormatFactoryPrivateData;
        unsafe { &(*private_data).runtime }
    }
}

unsafe extern "C" fn factory_get_ext_fn_wrapper(
    factory: &FFI_FileFormatFactory,
) -> SString {
    factory.inner().get_ext().into()
}

unsafe extern "C" fn factory_create_fn_wrapper(
    factory: &FFI_FileFormatFactory,
    session: FFI_SessionRef,
    format_options: SVec<(SString, SString)>,
) -> FFI_Result<FFI_FileFormat> {
    let runtime = factory.runtime().clone();
    let logical_codec = factory.logical_codec.clone();
    let internal_factory = Arc::clone(factory.inner());

    let mut foreign_session = None;
    let session = sresult_return!(
        session
            .as_local()
            .map(Ok::<&dyn Session, DataFusionError>)
            .unwrap_or_else(|| {
                foreign_session = Some(ForeignSession::try_from(&session)?);
                Ok(foreign_session.as_ref().unwrap())
            })
    );

    let format_options: HashMap<String, String> = format_options
        .into_iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();

    let file_format = sresult_return!(internal_factory.create(session, &format_options));

    FFI_Result::Ok(FFI_FileFormat::new_with_ffi_codec(
        file_format,
        runtime,
        logical_codec,
    ))
}

unsafe extern "C" fn factory_default_fn_wrapper(
    factory: &FFI_FileFormatFactory,
) -> FFI_FileFormat {
    let runtime = factory.runtime().clone();
    let logical_codec = factory.logical_codec.clone();
    let file_format = factory.inner().default();

    FFI_FileFormat::new_with_ffi_codec(file_format, runtime, logical_codec)
}

unsafe extern "C" fn factory_release_fn_wrapper(factory: &mut FFI_FileFormatFactory) {
    unsafe {
        debug_assert!(!factory.private_data.is_null());
        let private_data =
            Box::from_raw(factory.private_data as *mut FormatFactoryPrivateData);
        drop(private_data);
        factory.private_data = std::ptr::null_mut();
    }
}

unsafe extern "C" fn factory_clone_fn_wrapper(
    factory: &FFI_FileFormatFactory,
) -> FFI_FileFormatFactory {
    let runtime = factory.runtime().clone();
    let old_factory = Arc::clone(factory.inner());

    let private_data = Box::into_raw(Box::new(FormatFactoryPrivateData {
        factory: old_factory,
        runtime,
    })) as *mut c_void;

    FFI_FileFormatFactory {
        get_ext: factory_get_ext_fn_wrapper,
        create: factory_create_fn_wrapper,
        default: factory_default_fn_wrapper,
        logical_codec: factory.logical_codec.clone(),
        clone: factory_clone_fn_wrapper,
        release: factory_release_fn_wrapper,
        version: super::version,
        private_data,
        library_marker_id: crate::get_library_marker_id,
    }
}

impl Drop for FFI_FileFormatFactory {
    fn drop(&mut self) {
        unsafe { (self.release)(self) }
    }
}

impl Clone for FFI_FileFormatFactory {
    fn clone(&self) -> Self {
        unsafe { (self.clone)(self) }
    }
}

impl FFI_FileFormatFactory {
    /// Creates a new [`FFI_FileFormatFactory`].
    pub fn new(
        factory: Arc<dyn FileFormatFactory>,
        runtime: Option<Handle>,
        task_ctx_provider: impl Into<FFI_TaskContextProvider>,
        logical_codec: Option<Arc<dyn LogicalExtensionCodec>>,
    ) -> Self {
        let task_ctx_provider = task_ctx_provider.into();
        let logical_codec =
            logical_codec.unwrap_or_else(|| Arc::new(DefaultLogicalExtensionCodec {}));
        let logical_codec = FFI_LogicalExtensionCodec::new(
            logical_codec,
            runtime.clone(),
            task_ctx_provider,
        );
        Self::new_with_ffi_codec(factory, runtime, logical_codec)
    }

    /// Creates a new [`FFI_FileFormatFactory`] using an existing FFI logical
    /// codec. The [`FFI_FileFormat`] instances produced by [`Self::create`]
    /// and [`Self::default`] are given a default physical extension codec
    /// derived from `logical_codec`'s task context provider (see
    /// [`FFI_FileFormat::new_with_ffi_codec`] for the tradeoffs of this
    /// default).
    pub fn new_with_ffi_codec(
        factory: Arc<dyn FileFormatFactory>,
        runtime: Option<Handle>,
        logical_codec: FFI_LogicalExtensionCodec,
    ) -> Self {
        if let Some(factory) = factory.downcast_ref::<ForeignFileFormatFactory>() {
            return factory.0.clone();
        }
        let private_data = Box::new(FormatFactoryPrivateData { factory, runtime });

        Self {
            get_ext: factory_get_ext_fn_wrapper,
            create: factory_create_fn_wrapper,
            default: factory_default_fn_wrapper,
            logical_codec,
            clone: factory_clone_fn_wrapper,
            release: factory_release_fn_wrapper,
            version: super::version,
            private_data: Box::into_raw(private_data) as *mut c_void,
            library_marker_id: crate::get_library_marker_id,
        }
    }
}

/// This wrapper struct exists on the receiver side of the FFI interface, so
/// it has no guarantees about being able to access the data in
/// `private_data`. Any functions defined on this struct must only use the
/// stable functions provided in `FFI_FileFormatFactory` to interact with the
/// foreign file format factory.
#[derive(Debug)]
pub struct ForeignFileFormatFactory(pub FFI_FileFormatFactory);

unsafe impl Send for ForeignFileFormatFactory {}
unsafe impl Sync for ForeignFileFormatFactory {}

impl From<&FFI_FileFormatFactory> for Arc<dyn FileFormatFactory> {
    fn from(factory: &FFI_FileFormatFactory) -> Self {
        if (factory.library_marker_id)() == crate::get_library_marker_id() {
            Arc::clone(factory.inner())
        } else {
            Arc::new(ForeignFileFormatFactory(factory.clone()))
        }
    }
}

impl GetExt for ForeignFileFormatFactory {
    fn get_ext(&self) -> String {
        unsafe { (self.0.get_ext)(&self.0).into() }
    }
}

impl FileFormatFactory for ForeignFileFormatFactory {
    fn create(
        &self,
        state: &dyn Session,
        format_options: &HashMap<String, String>,
    ) -> Result<Arc<dyn FileFormat>> {
        let session = FFI_SessionRef::new(state, None, self.0.logical_codec.clone());
        let format_options: SVec<(SString, SString)> = format_options
            .iter()
            .map(|(k, v)| (SString::from(k.as_str()), SString::from(v.as_str())))
            .collect();

        let ffi_format =
            df_result!(unsafe { (self.0.create)(&self.0, session, format_options) })?;

        Ok((&ffi_format).into())
    }

    fn default(&self) -> Arc<dyn FileFormat> {
        let ffi_format = unsafe { (self.0.default)(&self.0) };
        (&ffi_format).into()
    }
}

/// A stable struct for sharing [`FileFormat`] across FFI boundaries.
#[repr(C)]
#[derive(Debug)]
pub struct FFI_FileFormat {
    /// Returns the default file extension for this format, e.g. "csv"
    get_ext: unsafe extern "C" fn(&Self) -> SString,

    /// Returns the extension for this format when compressed.
    get_ext_with_compression: unsafe extern "C" fn(
        &Self,
        compression: FFI_FileCompressionType,
    ) -> FFI_Result<SString>,

    /// Returns whether this instance uses compression, and if so, which kind.
    compression_type: unsafe extern "C" fn(&Self) -> FFI_Option<FFI_FileCompressionType>,

    /// Build the read [`ExecutionPlan`] for a `conf`, a prost-encoded
    /// `protobuf::FileScanExecConf`. See the module-level documentation on
    /// [`FfiPlanCodecAdapter`] for the encoding used for `conf`, and
    /// [`ForeignFileFormat::create_physical_plan`] for the caveat around
    /// [`FileSource`] reconstruction.
    create_physical_plan:
        unsafe extern "C" fn(
            &Self,
            session: FFI_SessionRef,
            conf: SVec<u8>,
            physical_codec: FFI_PhysicalExtensionCodec,
        ) -> FfiFuture<FFI_Result<FFI_ExecutionPlan>>,

    /// Build the write [`ExecutionPlan`] for a `conf`, a prost-encoded
    /// `protobuf::FileSinkConfig`.
    create_writer_physical_plan:
        unsafe extern "C" fn(
            &Self,
            input: FFI_ExecutionPlan,
            session: FFI_SessionRef,
            conf: SVec<u8>,
            order_requirements: FFI_Option<SVec<FFI_PhysicalSortExpr>>,
            physical_codec: FFI_PhysicalExtensionCodec,
        ) -> FfiFuture<FFI_Result<FFI_ExecutionPlan>>,

    /// Used to construct a [`FFI_SessionRef`] for the calls above.
    pub logical_codec: FFI_LogicalExtensionCodec,

    /// Used to (de)serialize the physical expressions embedded in
    /// `FileScanConfig`/`FileSinkConfig`.
    pub physical_codec: FFI_PhysicalExtensionCodec,

    /// Used to create a clone on the provider of the file format. This
    /// should only need to be called by the receiver of the file format.
    clone: unsafe extern "C" fn(format: &Self) -> Self,

    /// Release the memory of the private data when it is no longer being used.
    release: unsafe extern "C" fn(arg: &mut Self),

    /// Return the major DataFusion version number of this provider.
    pub version: unsafe extern "C" fn() -> u64,

    /// Internal data. This is only to be accessed by the provider of the
    /// file format. A [`ForeignFileFormat`] should never attempt to access
    /// this data.
    private_data: *mut c_void,

    /// Utility to identify when FFI objects are accessed locally through
    /// the foreign interface. See [`crate::get_library_marker_id`] and
    /// the crate's `README.md` for more information.
    pub library_marker_id: extern "C" fn() -> usize,
}

unsafe impl Send for FFI_FileFormat {}
unsafe impl Sync for FFI_FileFormat {}

struct FileFormatPrivateData {
    format: Arc<dyn FileFormat>,
    runtime: Option<Handle>,
}

impl FFI_FileFormat {
    fn inner(&self) -> &Arc<dyn FileFormat> {
        let private_data = self.private_data as *const FileFormatPrivateData;
        unsafe { &(*private_data).format }
    }

    fn runtime(&self) -> &Option<Handle> {
        let private_data = self.private_data as *const FileFormatPrivateData;
        unsafe { &(*private_data).runtime }
    }
}

unsafe extern "C" fn format_get_ext_fn_wrapper(format: &FFI_FileFormat) -> SString {
    format.inner().get_ext().into()
}

unsafe extern "C" fn format_get_ext_with_compression_fn_wrapper(
    format: &FFI_FileFormat,
    compression: FFI_FileCompressionType,
) -> FFI_Result<SString> {
    let compression: FileCompressionType = compression.into();
    format
        .inner()
        .get_ext_with_compression(&compression)
        .map(|ext| SString::from(ext.as_str()))
        .into()
}

unsafe extern "C" fn format_compression_type_fn_wrapper(
    format: &FFI_FileFormat,
) -> FFI_Option<FFI_FileCompressionType> {
    format
        .inner()
        .compression_type()
        .map(FFI_FileCompressionType::from)
        .into()
}

unsafe extern "C" fn format_create_physical_plan_fn_wrapper(
    format: &FFI_FileFormat,
    session: FFI_SessionRef,
    conf: SVec<u8>,
    physical_codec: FFI_PhysicalExtensionCodec,
) -> FfiFuture<FFI_Result<FFI_ExecutionPlan>> {
    let runtime = format.runtime().clone();
    let internal_format = Arc::clone(format.inner());

    async move {
        let mut foreign_session = None;
        let session = sresult_return!(
            session
                .as_local()
                .map(Ok::<&dyn Session, DataFusionError>)
                .unwrap_or_else(|| {
                    foreign_session = Some(ForeignSession::try_from(&session)?);
                    Ok(foreign_session.as_ref().unwrap())
                })
        );

        let codec: Arc<dyn PhysicalExtensionCodec> = (&physical_codec).into();
        let task_ctx: Arc<TaskContext> =
            sresult_return!((&physical_codec.task_ctx_provider).try_into());

        let proto_conf = sresult_return!(
            protobuf::FileScanExecConf::decode(conf.as_slice()).map_err(|e| {
                internal_datafusion_err!("failed to decode FileScanExecConf: {e}")
            })
        );

        // The file source is rebuilt locally (on the provider side, which
        // owns the real `FileFormat`) from the table schema embedded in the
        // proto payload. This sidesteps needing a `FileSource` FFI wrapper,
        // but does mean a caller that needs to call `file_source()` itself
        // (e.g. a `ListingTable`-style flow building its own
        // `FileScanConfig`) cannot do so through a foreign `FileFormat` yet;
        // direct use of `create_physical_plan` with a `FileScanConfig`
        // obtained some other way still works.
        let table_schema =
            sresult_return!(FileScanConfig::parse_table_schema_from_proto(&proto_conf));
        let file_source = internal_format.file_source(table_schema);

        // Scoped so the non-`Send` `ExecutionPlanDecodeCtx` (it borrows a
        // `&dyn ExecutionPlanDecode` trait object with no `Send`/`Sync`
        // bound) is dropped before the `.await` below.
        let scan_conf = {
            let adapter = FfiPlanCodecAdapter {
                codec: codec.as_ref(),
                task_ctx: task_ctx.as_ref(),
            };
            let decode_ctx = ExecutionPlanDecodeCtx::new(&adapter);

            sresult_return!(FileScanConfig::try_from_proto(
                &proto_conf,
                &decode_ctx,
                file_source
            ))
        };

        let plan = sresult_return!(
            internal_format
                .create_physical_plan(session, scan_conf)
                .await
        );

        FFI_Result::Ok(FFI_ExecutionPlan::new(plan, runtime.clone()))
    }
    .into_ffi()
}

unsafe extern "C" fn format_create_writer_physical_plan_fn_wrapper(
    format: &FFI_FileFormat,
    input: FFI_ExecutionPlan,
    session: FFI_SessionRef,
    conf: SVec<u8>,
    order_requirements: FFI_Option<SVec<FFI_PhysicalSortExpr>>,
    // `FileSinkConfig`'s proto encoding carries no expressions, so decoding it
    // needs no extension codec; kept for signature symmetry with
    // `create_physical_plan` and forward-compatibility.
    _physical_codec: FFI_PhysicalExtensionCodec,
) -> FfiFuture<FFI_Result<FFI_ExecutionPlan>> {
    let runtime = format.runtime().clone();
    let internal_format = Arc::clone(format.inner());

    async move {
        let mut foreign_session = None;
        let session = sresult_return!(
            session
                .as_local()
                .map(Ok::<&dyn Session, DataFusionError>)
                .unwrap_or_else(|| {
                    foreign_session = Some(ForeignSession::try_from(&session)?);
                    Ok(foreign_session.as_ref().unwrap())
                })
        );

        let input = sresult_return!(<Arc<dyn ExecutionPlan>>::try_from(&input));

        let proto_conf =
            sresult_return!(ProtoFileSinkConfig::decode(conf.as_slice()).map_err(|e| {
                internal_datafusion_err!("failed to decode FileSinkConfig: {e}")
            }));
        let sink_conf: FileSinkConfig = sresult_return!((&proto_conf).try_into());

        let order_requirements: Option<LexRequirement> =
            order_requirements.into_option().and_then(
                |exprs: SVec<FFI_PhysicalSortExpr>| {
                    LexRequirement::new(exprs.iter().map(|e| {
                        PhysicalSortRequirement::from(PhysicalSortExpr::from(e))
                    }))
                },
            );

        let plan = sresult_return!(
            internal_format
                .create_writer_physical_plan(
                    input,
                    session,
                    sink_conf,
                    order_requirements
                )
                .await
        );

        FFI_Result::Ok(FFI_ExecutionPlan::new(plan, runtime.clone()))
    }
    .into_ffi()
}

unsafe extern "C" fn format_release_fn_wrapper(format: &mut FFI_FileFormat) {
    unsafe {
        debug_assert!(!format.private_data.is_null());
        let private_data =
            Box::from_raw(format.private_data as *mut FileFormatPrivateData);
        drop(private_data);
        format.private_data = std::ptr::null_mut();
    }
}

unsafe extern "C" fn format_clone_fn_wrapper(format: &FFI_FileFormat) -> FFI_FileFormat {
    let runtime = format.runtime().clone();
    let old_format = Arc::clone(format.inner());

    let private_data = Box::into_raw(Box::new(FileFormatPrivateData {
        format: old_format,
        runtime,
    })) as *mut c_void;

    FFI_FileFormat {
        get_ext: format_get_ext_fn_wrapper,
        get_ext_with_compression: format_get_ext_with_compression_fn_wrapper,
        compression_type: format_compression_type_fn_wrapper,
        create_physical_plan: format_create_physical_plan_fn_wrapper,
        create_writer_physical_plan: format_create_writer_physical_plan_fn_wrapper,
        logical_codec: format.logical_codec.clone(),
        physical_codec: format.physical_codec.clone(),
        clone: format_clone_fn_wrapper,
        release: format_release_fn_wrapper,
        version: super::version,
        private_data,
        library_marker_id: crate::get_library_marker_id,
    }
}

impl Drop for FFI_FileFormat {
    fn drop(&mut self) {
        unsafe { (self.release)(self) }
    }
}

impl Clone for FFI_FileFormat {
    fn clone(&self) -> Self {
        unsafe { (self.clone)(self) }
    }
}

impl FFI_FileFormat {
    /// Creates a new [`FFI_FileFormat`] using an existing FFI logical codec.
    ///
    /// The physical extension codec used to (de)serialize the physical
    /// expressions embedded in `FileScanConfig`/`FileSinkConfig` is
    /// synthesized as a [`DefaultPhysicalExtensionCodec`] reusing
    /// `logical_codec`'s task context provider (mirroring
    /// [`FFI_SessionRef::new`]'s tradeoffs: this does not successfully
    /// decode custom physical extension nodes). Use
    /// [`Self::new_with_ffi_codecs`] with an explicitly configured physical
    /// codec when custom physical nodes must cross the FFI boundary.
    pub fn new_with_ffi_codec(
        format: Arc<dyn FileFormat>,
        runtime: Option<Handle>,
        logical_codec: FFI_LogicalExtensionCodec,
    ) -> Self {
        let physical_codec = FFI_PhysicalExtensionCodec::new(
            Arc::new(DefaultPhysicalExtensionCodec {}),
            runtime.clone(),
            logical_codec.task_ctx_provider.clone(),
        );
        Self::new_with_ffi_codecs(format, runtime, logical_codec, physical_codec)
    }

    /// Creates a new [`FFI_FileFormat`] using existing FFI logical and
    /// physical codecs.
    pub fn new_with_ffi_codecs(
        format: Arc<dyn FileFormat>,
        runtime: Option<Handle>,
        logical_codec: FFI_LogicalExtensionCodec,
        physical_codec: FFI_PhysicalExtensionCodec,
    ) -> Self {
        if let Some(format) = format.downcast_ref::<ForeignFileFormat>() {
            return format.0.clone();
        }
        let private_data = Box::new(FileFormatPrivateData { format, runtime });

        Self {
            get_ext: format_get_ext_fn_wrapper,
            get_ext_with_compression: format_get_ext_with_compression_fn_wrapper,
            compression_type: format_compression_type_fn_wrapper,
            create_physical_plan: format_create_physical_plan_fn_wrapper,
            create_writer_physical_plan: format_create_writer_physical_plan_fn_wrapper,
            logical_codec,
            physical_codec,
            clone: format_clone_fn_wrapper,
            release: format_release_fn_wrapper,
            version: super::version,
            private_data: Box::into_raw(private_data) as *mut c_void,
            library_marker_id: crate::get_library_marker_id,
        }
    }
}

/// This wrapper struct exists on the receiver side of the FFI interface, so
/// it has no guarantees about being able to access the data in
/// `private_data`. Any functions defined on this struct must only use the
/// stable functions provided in `FFI_FileFormat` to interact with the
/// foreign file format.
#[derive(Debug)]
pub struct ForeignFileFormat(pub FFI_FileFormat);

unsafe impl Send for ForeignFileFormat {}
unsafe impl Sync for ForeignFileFormat {}

impl From<&FFI_FileFormat> for Arc<dyn FileFormat> {
    fn from(format: &FFI_FileFormat) -> Self {
        if (format.library_marker_id)() == crate::get_library_marker_id() {
            Arc::clone(format.inner())
        } else {
            Arc::new(ForeignFileFormat(format.clone()))
        }
    }
}

#[async_trait]
impl FileFormat for ForeignFileFormat {
    fn get_ext(&self) -> String {
        unsafe { (self.0.get_ext)(&self.0).into() }
    }

    fn get_ext_with_compression(
        &self,
        file_compression_type: &FileCompressionType,
    ) -> Result<String> {
        let compression: FFI_FileCompressionType = (*file_compression_type).into();
        let result = unsafe { (self.0.get_ext_with_compression)(&self.0, compression) };
        df_result!(result).map(|ext| ext.to_string())
    }

    fn compression_type(&self) -> Option<FileCompressionType> {
        let ffi_opt = unsafe { (self.0.compression_type)(&self.0) };
        ffi_opt.into_option().map(FileCompressionType::from)
    }

    async fn infer_schema(
        &self,
        _state: &dyn Session,
        _store: &Arc<dyn ObjectStore>,
        _objects: &[ObjectMeta],
    ) -> Result<SchemaRef> {
        not_impl_err!(
            "FFI FileFormat does not support infer_schema: it requires a live \
             ObjectStore, and there is no FFI_ObjectStore wrapper yet"
        )
    }

    async fn infer_stats(
        &self,
        _state: &dyn Session,
        _store: &Arc<dyn ObjectStore>,
        _table_schema: SchemaRef,
        _object: &ObjectMeta,
    ) -> Result<Statistics> {
        not_impl_err!(
            "FFI FileFormat does not support infer_stats: it requires a live \
             ObjectStore, and there is no FFI_ObjectStore wrapper yet"
        )
    }

    async fn infer_ordering(
        &self,
        _state: &dyn Session,
        _store: &Arc<dyn ObjectStore>,
        _table_schema: SchemaRef,
        _object: &ObjectMeta,
    ) -> Result<Option<LexOrdering>> {
        not_impl_err!(
            "FFI FileFormat does not support infer_ordering: it requires a live \
             ObjectStore, and there is no FFI_ObjectStore wrapper yet"
        )
    }

    async fn create_physical_plan(
        &self,
        state: &dyn Session,
        conf: FileScanConfig,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let session = FFI_SessionRef::new(state, None, self.0.logical_codec.clone());

        let codec: Arc<dyn PhysicalExtensionCodec> = (&self.0.physical_codec).into();
        let task_ctx: Arc<TaskContext> =
            (&self.0.physical_codec.task_ctx_provider).try_into()?;

        // Scoped so the non-`Send` `ExecutionPlanEncodeCtx` (it borrows a
        // `&dyn ExecutionPlanEncode` trait object with no `Send`/`Sync`
        // bound) is dropped before the `.await` below.
        let proto_conf = {
            let adapter = FfiPlanCodecAdapter {
                codec: codec.as_ref(),
                task_ctx: task_ctx.as_ref(),
            };
            let encode_ctx = ExecutionPlanEncodeCtx::new(&adapter);

            conf.try_to_proto(&encode_ctx)?
        };
        let bytes = proto_conf.encode_to_vec();

        let plan = unsafe {
            let maybe_plan = (self.0.create_physical_plan)(
                &self.0,
                session,
                bytes.into_iter().collect(),
                self.0.physical_codec.clone(),
            )
            .await;

            <Arc<dyn ExecutionPlan>>::try_from(&df_result!(maybe_plan)?)?
        };

        Ok(plan)
    }

    async fn create_writer_physical_plan(
        &self,
        input: Arc<dyn ExecutionPlan>,
        state: &dyn Session,
        conf: FileSinkConfig,
        order_requirements: Option<LexRequirement>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let session = FFI_SessionRef::new(state, None, self.0.logical_codec.clone());

        let rc = Handle::try_current().ok();
        let input = FFI_ExecutionPlan::new(input, rc);

        let proto_conf: ProtoFileSinkConfig = (&conf).try_into()?;
        let bytes = proto_conf.encode_to_vec();

        let order_requirements: FFI_Option<SVec<FFI_PhysicalSortExpr>> =
            order_requirements
                .map(|reqs| {
                    reqs.into_iter()
                        .map(|req| {
                            FFI_PhysicalSortExpr::from(&PhysicalSortExpr::from(req))
                        })
                        .collect()
                })
                .into();

        let plan = unsafe {
            let maybe_plan = (self.0.create_writer_physical_plan)(
                &self.0,
                input,
                session,
                bytes.into_iter().collect(),
                order_requirements,
                self.0.physical_codec.clone(),
            )
            .await;

            <Arc<dyn ExecutionPlan>>::try_from(&df_result!(maybe_plan)?)?
        };

        Ok(plan)
    }

    /// Not supported: constructing a real [`FileSource`] requires resolving
    /// data through a live `ObjectStore`, and there is no `FFI_ObjectStore`
    /// wrapper yet (see the module-level FFI file format documentation).
    /// Unlike the other unsupported methods above, this one is infallible in
    /// the [`FileFormat`] trait, so it cannot return a `not_impl_err!`;
    /// panicking is the only way to signal the same "not yet supported"
    /// condition. Build the `FileScanConfig`'s `file_source` locally (via
    /// the real, non-FFI `FileFormat`) and call
    /// [`Self::create_physical_plan`] directly instead of going through this
    /// method on a foreign `FileFormat`.
    fn file_source(&self, _table_schema: TableSchema) -> Arc<dyn FileSource> {
        panic!(
            "ForeignFileFormat::file_source is not supported: FFI FileFormat has no \
             ObjectStore wrapper yet, so a foreign FileFormat cannot construct its own \
             FileSource. Build the FileScanConfig's file_source locally (via the real, \
             non-FFI FileFormat) and call create_physical_plan directly instead."
        )
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{Int64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use datafusion::datasource::memory::MemorySourceConfig;
    use datafusion::prelude::SessionContext;
    use datafusion_datasource::file_groups::FileGroup;
    use datafusion_datasource::file_sink_config::FileOutputMode;
    use datafusion_datasource_csv::file_format::{CsvFormat, CsvFormatFactory};
    use datafusion_execution::TaskContextProvider;
    use datafusion_execution::object_store::ObjectStoreUrl;
    use datafusion_expr::dml::InsertOp;
    use datafusion_physical_plan::collect;
    use object_store::local::LocalFileSystem;

    use super::*;

    fn test_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("b", DataType::Utf8, false),
        ]))
    }

    #[test]
    fn test_ffi_file_format_factory_get_ext() {
        let factory = Arc::new(CsvFormatFactory::new()) as Arc<dyn FileFormatFactory>;
        let (_ctx, task_ctx_provider) = crate::util::tests::test_session_and_ctx();

        let mut ffi_factory =
            FFI_FileFormatFactory::new(factory, None, task_ctx_provider, None);
        ffi_factory.library_marker_id = crate::mock_foreign_marker_id;

        let foreign_factory: Arc<dyn FileFormatFactory> = (&ffi_factory).into();
        assert_eq!(foreign_factory.get_ext(), "csv");
    }

    #[test]
    fn test_ffi_file_format_create_round_trip() -> Result<()> {
        let factory = Arc::new(CsvFormatFactory::new()) as Arc<dyn FileFormatFactory>;
        let (ctx, task_ctx_provider) = crate::util::tests::test_session_and_ctx();

        let mut ffi_factory =
            FFI_FileFormatFactory::new(factory, None, task_ctx_provider, None);
        ffi_factory.library_marker_id = crate::mock_foreign_marker_id;

        let foreign_factory: Arc<dyn FileFormatFactory> = (&ffi_factory).into();

        let mut format_options = HashMap::new();
        format_options.insert("format.delimiter".to_string(), ";".to_string());

        let state = ctx.state();
        let format = foreign_factory.create(&state, &format_options)?;

        assert_eq!(format.get_ext(), "csv");
        assert_eq!(
            format.get_ext_with_compression(&FileCompressionType::UNCOMPRESSED)?,
            "csv"
        );
        assert_eq!(
            format.get_ext_with_compression(&FileCompressionType::GZIP)?,
            "csv.gz"
        );
        assert_eq!(
            format.compression_type(),
            Some(FileCompressionType::UNCOMPRESSED)
        );

        let default_format = foreign_factory.default();
        assert_eq!(default_format.get_ext(), "csv");

        Ok(())
    }

    fn input_plan_for(schema: &Arc<Schema>) -> Result<Arc<dyn ExecutionPlan>> {
        let batch = RecordBatch::try_new(
            Arc::clone(schema),
            vec![
                Arc::new(Int64Array::from(vec![1, 2, 3])),
                Arc::new(StringArray::from(vec!["x", "y", "z"])),
            ],
        )?;

        let source =
            MemorySourceConfig::try_new_exec(&[vec![batch]], Arc::clone(schema), None)?;
        Ok(source as Arc<dyn ExecutionPlan>)
    }

    #[tokio::test]
    async fn test_ffi_file_format_create_writer_physical_plan_round_trip() -> Result<()> {
        let tmp_dir = tempfile::tempdir().expect("failed to create temp dir");
        let object_store_url = ObjectStoreUrl::local_filesystem();

        let ctx = Arc::new(SessionContext::new());
        ctx.runtime_env().register_object_store(
            object_store_url.as_ref(),
            Arc::new(LocalFileSystem::new()),
        );

        let schema = test_schema();

        let format = Arc::new(CsvFormat::default()) as Arc<dyn FileFormat>;
        let task_ctx_provider = Arc::clone(&ctx) as Arc<dyn TaskContextProvider>;
        let task_ctx_provider = FFI_TaskContextProvider::from(&task_ctx_provider);
        let logical_codec = FFI_LogicalExtensionCodec::new(
            Arc::new(DefaultLogicalExtensionCodec {}),
            None,
            task_ctx_provider,
        );
        let mut ffi_format =
            FFI_FileFormat::new_with_ffi_codec(format, None, logical_codec);
        ffi_format.library_marker_id = crate::mock_foreign_marker_id;

        let foreign_format: Arc<dyn FileFormat> = (&ffi_format).into();

        let output_path = tmp_dir.path().join("out.csv");
        let sink_conf = FileSinkConfig {
            original_url: String::new(),
            object_store_url: object_store_url.clone(),
            file_group: FileGroup::new(vec![]),
            table_paths: vec![datafusion_datasource::ListingTableUrl::parse(
                output_path.to_str().unwrap(),
            )?],
            output_schema: Arc::clone(&schema),
            table_partition_cols: vec![],
            insert_op: InsertOp::Append,
            keep_partition_by_columns: false,
            file_extension: "csv".to_string(),
            file_output_mode: FileOutputMode::SingleFile,
        };

        let input = input_plan_for(&schema)?;
        let state = ctx.state();

        let write_plan = foreign_format
            .create_writer_physical_plan(input, &state, sink_conf, None)
            .await?;

        let batches = collect(write_plan, ctx.task_ctx()).await?;
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].num_rows(), 1);

        let written = std::fs::read_to_string(&output_path)
            .expect("expected output CSV file to have been written");
        assert!(written.contains("1,x"));
        assert!(written.contains("2,y"));
        assert!(written.contains("3,z"));

        Ok(())
    }

    #[tokio::test]
    async fn test_ffi_file_format_create_physical_plan_round_trip() -> Result<()> {
        let tmp_dir = tempfile::tempdir().expect("failed to create temp dir");
        let object_store_url = ObjectStoreUrl::local_filesystem();

        let ctx = Arc::new(SessionContext::new());
        ctx.runtime_env().register_object_store(
            object_store_url.as_ref(),
            Arc::new(LocalFileSystem::new()),
        );

        let schema = test_schema();
        let input_path = tmp_dir.path().join("in.csv");
        std::fs::write(&input_path, "1,x\n2,y\n3,z\n")
            .expect("failed to write input CSV file");

        let real_format = CsvFormat::default().with_has_header(false);
        let table_schema = TableSchema::from(Arc::clone(&schema));
        let file_source = real_format.file_source(table_schema);

        let object_meta = ObjectMeta {
            location: object_store::path::Path::from_filesystem_path(&input_path)
                .expect("valid path"),
            last_modified: chrono::Utc::now(),
            size: std::fs::metadata(&input_path)?.len(),
            e_tag: None,
            version: None,
        };
        let partitioned_file = datafusion_datasource::PartitionedFile::from(object_meta);
        let file_group = FileGroup::new(vec![partitioned_file]);

        let conf = datafusion_datasource::file_scan_config::FileScanConfigBuilder::new(
            object_store_url.clone(),
            file_source,
        )
        .with_file_group(file_group)
        .build();

        let task_ctx_provider = Arc::clone(&ctx) as Arc<dyn TaskContextProvider>;
        let task_ctx_provider = FFI_TaskContextProvider::from(&task_ctx_provider);
        let logical_codec = FFI_LogicalExtensionCodec::new(
            Arc::new(DefaultLogicalExtensionCodec {}),
            None,
            task_ctx_provider,
        );
        let format = Arc::new(real_format) as Arc<dyn FileFormat>;
        let mut ffi_format =
            FFI_FileFormat::new_with_ffi_codec(format, None, logical_codec);
        ffi_format.library_marker_id = crate::mock_foreign_marker_id;

        let foreign_format: Arc<dyn FileFormat> = (&ffi_format).into();

        let state = ctx.state();
        let plan = foreign_format.create_physical_plan(&state, conf).await?;

        let batches = collect(plan, ctx.task_ctx()).await?;
        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 3);

        Ok(())
    }

    #[test]
    fn test_ffi_file_format_local_bypass() {
        let format = Arc::new(CsvFormat::default()) as Arc<dyn FileFormat>;
        let (_ctx, task_ctx_provider) = crate::util::tests::test_session_and_ctx();
        let logical_codec = FFI_LogicalExtensionCodec::new(
            Arc::new(DefaultLogicalExtensionCodec {}),
            None,
            task_ctx_provider,
        );

        let ffi_format = FFI_FileFormat::new_with_ffi_codec(format, None, logical_codec);

        let foreign_format: Arc<dyn FileFormat> = (&ffi_format).into();
        assert!(foreign_format.downcast_ref::<CsvFormat>().is_some());

        let mut ffi_format = ffi_format;
        ffi_format.library_marker_id = crate::mock_foreign_marker_id;
        let foreign_format: Arc<dyn FileFormat> = (&ffi_format).into();
        assert!(foreign_format.downcast_ref::<ForeignFileFormat>().is_some());
    }

    #[test]
    #[should_panic(expected = "file_source is not supported")]
    fn test_ffi_file_format_file_source_panics() {
        let format = Arc::new(CsvFormat::default()) as Arc<dyn FileFormat>;
        let (_ctx, task_ctx_provider) = crate::util::tests::test_session_and_ctx();
        let logical_codec = FFI_LogicalExtensionCodec::new(
            Arc::new(DefaultLogicalExtensionCodec {}),
            None,
            task_ctx_provider,
        );
        let mut ffi_format =
            FFI_FileFormat::new_with_ffi_codec(format, None, logical_codec);
        ffi_format.library_marker_id = crate::mock_foreign_marker_id;

        let foreign_format: Arc<dyn FileFormat> = (&ffi_format).into();
        let table_schema = TableSchema::from(test_schema());
        let _ = foreign_format.file_source(table_schema);
    }
}
