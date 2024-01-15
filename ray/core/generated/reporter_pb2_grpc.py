# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from . import reporter_pb2 as src_dot_ray_dot_protobuf_dot_reporter__pb2


class ReporterServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.GetProfilingStats = channel.unary_unary(
                '/ray.rpc.ReporterService/GetProfilingStats',
                request_serializer=src_dot_ray_dot_protobuf_dot_reporter__pb2.GetProfilingStatsRequest.SerializeToString,
                response_deserializer=src_dot_ray_dot_protobuf_dot_reporter__pb2.GetProfilingStatsReply.FromString,
                )
        self.ReportMetrics = channel.unary_unary(
                '/ray.rpc.ReporterService/ReportMetrics',
                request_serializer=src_dot_ray_dot_protobuf_dot_reporter__pb2.ReportMetricsRequest.SerializeToString,
                response_deserializer=src_dot_ray_dot_protobuf_dot_reporter__pb2.ReportMetricsReply.FromString,
                )
        self.ReportOCMetrics = channel.unary_unary(
                '/ray.rpc.ReporterService/ReportOCMetrics',
                request_serializer=src_dot_ray_dot_protobuf_dot_reporter__pb2.ReportOCMetricsRequest.SerializeToString,
                response_deserializer=src_dot_ray_dot_protobuf_dot_reporter__pb2.ReportOCMetricsReply.FromString,
                )
        self.GetTraceback = channel.unary_unary(
                '/ray.rpc.ReporterService/GetTraceback',
                request_serializer=src_dot_ray_dot_protobuf_dot_reporter__pb2.GetTracebackRequest.SerializeToString,
                response_deserializer=src_dot_ray_dot_protobuf_dot_reporter__pb2.GetTracebackReply.FromString,
                )
        self.CpuProfiling = channel.unary_unary(
                '/ray.rpc.ReporterService/CpuProfiling',
                request_serializer=src_dot_ray_dot_protobuf_dot_reporter__pb2.CpuProfilingRequest.SerializeToString,
                response_deserializer=src_dot_ray_dot_protobuf_dot_reporter__pb2.CpuProfilingReply.FromString,
                )
        self.MemoryProfiling = channel.unary_unary(
                '/ray.rpc.ReporterService/MemoryProfiling',
                request_serializer=src_dot_ray_dot_protobuf_dot_reporter__pb2.MemoryProfilingRequest.SerializeToString,
                response_deserializer=src_dot_ray_dot_protobuf_dot_reporter__pb2.MemoryProfilingReply.FromString,
                )


class ReporterServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def GetProfilingStats(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def ReportMetrics(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def ReportOCMetrics(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetTraceback(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def CpuProfiling(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def MemoryProfiling(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_ReporterServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'GetProfilingStats': grpc.unary_unary_rpc_method_handler(
                    servicer.GetProfilingStats,
                    request_deserializer=src_dot_ray_dot_protobuf_dot_reporter__pb2.GetProfilingStatsRequest.FromString,
                    response_serializer=src_dot_ray_dot_protobuf_dot_reporter__pb2.GetProfilingStatsReply.SerializeToString,
            ),
            'ReportMetrics': grpc.unary_unary_rpc_method_handler(
                    servicer.ReportMetrics,
                    request_deserializer=src_dot_ray_dot_protobuf_dot_reporter__pb2.ReportMetricsRequest.FromString,
                    response_serializer=src_dot_ray_dot_protobuf_dot_reporter__pb2.ReportMetricsReply.SerializeToString,
            ),
            'ReportOCMetrics': grpc.unary_unary_rpc_method_handler(
                    servicer.ReportOCMetrics,
                    request_deserializer=src_dot_ray_dot_protobuf_dot_reporter__pb2.ReportOCMetricsRequest.FromString,
                    response_serializer=src_dot_ray_dot_protobuf_dot_reporter__pb2.ReportOCMetricsReply.SerializeToString,
            ),
            'GetTraceback': grpc.unary_unary_rpc_method_handler(
                    servicer.GetTraceback,
                    request_deserializer=src_dot_ray_dot_protobuf_dot_reporter__pb2.GetTracebackRequest.FromString,
                    response_serializer=src_dot_ray_dot_protobuf_dot_reporter__pb2.GetTracebackReply.SerializeToString,
            ),
            'CpuProfiling': grpc.unary_unary_rpc_method_handler(
                    servicer.CpuProfiling,
                    request_deserializer=src_dot_ray_dot_protobuf_dot_reporter__pb2.CpuProfilingRequest.FromString,
                    response_serializer=src_dot_ray_dot_protobuf_dot_reporter__pb2.CpuProfilingReply.SerializeToString,
            ),
            'MemoryProfiling': grpc.unary_unary_rpc_method_handler(
                    servicer.MemoryProfiling,
                    request_deserializer=src_dot_ray_dot_protobuf_dot_reporter__pb2.MemoryProfilingRequest.FromString,
                    response_serializer=src_dot_ray_dot_protobuf_dot_reporter__pb2.MemoryProfilingReply.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'ray.rpc.ReporterService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class ReporterService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def GetProfilingStats(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.rpc.ReporterService/GetProfilingStats',
            src_dot_ray_dot_protobuf_dot_reporter__pb2.GetProfilingStatsRequest.SerializeToString,
            src_dot_ray_dot_protobuf_dot_reporter__pb2.GetProfilingStatsReply.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def ReportMetrics(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.rpc.ReporterService/ReportMetrics',
            src_dot_ray_dot_protobuf_dot_reporter__pb2.ReportMetricsRequest.SerializeToString,
            src_dot_ray_dot_protobuf_dot_reporter__pb2.ReportMetricsReply.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def ReportOCMetrics(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.rpc.ReporterService/ReportOCMetrics',
            src_dot_ray_dot_protobuf_dot_reporter__pb2.ReportOCMetricsRequest.SerializeToString,
            src_dot_ray_dot_protobuf_dot_reporter__pb2.ReportOCMetricsReply.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetTraceback(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.rpc.ReporterService/GetTraceback',
            src_dot_ray_dot_protobuf_dot_reporter__pb2.GetTracebackRequest.SerializeToString,
            src_dot_ray_dot_protobuf_dot_reporter__pb2.GetTracebackReply.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def CpuProfiling(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.rpc.ReporterService/CpuProfiling',
            src_dot_ray_dot_protobuf_dot_reporter__pb2.CpuProfilingRequest.SerializeToString,
            src_dot_ray_dot_protobuf_dot_reporter__pb2.CpuProfilingReply.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def MemoryProfiling(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.rpc.ReporterService/MemoryProfiling',
            src_dot_ray_dot_protobuf_dot_reporter__pb2.MemoryProfilingRequest.SerializeToString,
            src_dot_ray_dot_protobuf_dot_reporter__pb2.MemoryProfilingReply.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)


class LogServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.ListLogs = channel.unary_unary(
                '/ray.rpc.LogService/ListLogs',
                request_serializer=src_dot_ray_dot_protobuf_dot_reporter__pb2.ListLogsRequest.SerializeToString,
                response_deserializer=src_dot_ray_dot_protobuf_dot_reporter__pb2.ListLogsReply.FromString,
                )
        self.StreamLog = channel.unary_stream(
                '/ray.rpc.LogService/StreamLog',
                request_serializer=src_dot_ray_dot_protobuf_dot_reporter__pb2.StreamLogRequest.SerializeToString,
                response_deserializer=src_dot_ray_dot_protobuf_dot_reporter__pb2.StreamLogReply.FromString,
                )


class LogServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def ListLogs(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def StreamLog(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_LogServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'ListLogs': grpc.unary_unary_rpc_method_handler(
                    servicer.ListLogs,
                    request_deserializer=src_dot_ray_dot_protobuf_dot_reporter__pb2.ListLogsRequest.FromString,
                    response_serializer=src_dot_ray_dot_protobuf_dot_reporter__pb2.ListLogsReply.SerializeToString,
            ),
            'StreamLog': grpc.unary_stream_rpc_method_handler(
                    servicer.StreamLog,
                    request_deserializer=src_dot_ray_dot_protobuf_dot_reporter__pb2.StreamLogRequest.FromString,
                    response_serializer=src_dot_ray_dot_protobuf_dot_reporter__pb2.StreamLogReply.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'ray.rpc.LogService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class LogService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def ListLogs(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ray.rpc.LogService/ListLogs',
            src_dot_ray_dot_protobuf_dot_reporter__pb2.ListLogsRequest.SerializeToString,
            src_dot_ray_dot_protobuf_dot_reporter__pb2.ListLogsReply.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def StreamLog(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_stream(request, target, '/ray.rpc.LogService/StreamLog',
            src_dot_ray_dot_protobuf_dot_reporter__pb2.StreamLogRequest.SerializeToString,
            src_dot_ray_dot_protobuf_dot_reporter__pb2.StreamLogReply.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)