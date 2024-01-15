# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: src/ray/protobuf/ray_client.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n!src/ray/protobuf/ray_client.proto\x12\x07ray.rpc\"\xb5\x01\n\x03\x41rg\x12+\n\x05local\x18\x01 \x01(\x0e\x32\x15.ray.rpc.Arg.LocalityR\x05local\x12!\n\x0creference_id\x18\x02 \x01(\x0cR\x0breferenceId\x12\x12\n\x04\x64\x61ta\x18\x03 \x01(\x0cR\x04\x64\x61ta\x12!\n\x04type\x18\x04 \x01(\x0e\x32\r.ray.rpc.TypeR\x04type\"\'\n\x08Locality\x12\x0c\n\x08INTERNED\x10\x00\x12\r\n\tREFERENCE\x10\x01\"6\n\x0bTaskOptions\x12\'\n\x0fpickled_options\x18\x01 \x01(\x0cR\x0epickledOptions\"\xf4\x04\n\nClientTask\x12\x36\n\x04type\x18\x01 \x01(\x0e\x32\".ray.rpc.ClientTask.RemoteExecTypeR\x04type\x12\x12\n\x04name\x18\x02 \x01(\tR\x04name\x12\x1d\n\npayload_id\x18\x03 \x01(\x0cR\tpayloadId\x12 \n\x04\x61rgs\x18\x04 \x03(\x0b\x32\x0c.ray.rpc.ArgR\x04\x61rgs\x12\x37\n\x06kwargs\x18\x05 \x03(\x0b\x32\x1f.ray.rpc.ClientTask.KwargsEntryR\x06kwargs\x12\x1b\n\tclient_id\x18\x06 \x01(\tR\x08\x63lientId\x12.\n\x07options\x18\x07 \x01(\x0b\x32\x14.ray.rpc.TaskOptionsR\x07options\x12?\n\x10\x62\x61seline_options\x18\x08 \x01(\x0b\x32\x14.ray.rpc.TaskOptionsR\x0f\x62\x61selineOptions\x12\x1c\n\tnamespace\x18\t \x01(\tR\tnamespace\x12\x12\n\x04\x64\x61ta\x18\n \x01(\x0cR\x04\x64\x61ta\x12\x19\n\x08\x63hunk_id\x18\x0b \x01(\x05R\x07\x63hunkId\x12!\n\x0ctotal_chunks\x18\x0c \x01(\x05R\x0btotalChunks\x1aG\n\x0bKwargsEntry\x12\x10\n\x03key\x18\x01 \x01(\tR\x03key\x12\"\n\x05value\x18\x02 \x01(\x0b\x32\x0c.ray.rpc.ArgR\x05value:\x02\x38\x01\"Y\n\x0eRemoteExecType\x12\x0c\n\x08\x46UNCTION\x10\x00\x12\t\n\x05\x41\x43TOR\x10\x01\x12\n\n\x06METHOD\x10\x02\x12\x11\n\rSTATIC_METHOD\x10\x03\x12\x0f\n\x0bNAMED_ACTOR\x10\x04\"]\n\x10\x43lientTaskTicket\x12\x14\n\x05valid\x18\x01 \x01(\x08R\x05valid\x12\x1d\n\nreturn_ids\x18\x02 \x03(\x0cR\treturnIds\x12\x14\n\x05\x65rror\x18\x03 \x01(\x0cR\x05\x65rror\"\xbc\x01\n\nPutRequest\x12\x12\n\x04\x64\x61ta\x18\x01 \x01(\x0cR\x04\x64\x61ta\x12\"\n\rclient_ref_id\x18\x02 \x01(\x0cR\x0b\x63lientRefId\x12\x19\n\x08\x63hunk_id\x18\x03 \x01(\x05R\x07\x63hunkId\x12!\n\x0ctotal_chunks\x18\x04 \x01(\x05R\x0btotalChunks\x12\x1d\n\ntotal_size\x18\x05 \x01(\x03R\ttotalSize\x12\x19\n\x08owner_id\x18\x06 \x01(\x0cR\x07ownerId\"I\n\x0bPutResponse\x12\x0e\n\x02id\x18\x01 \x01(\x0cR\x02id\x12\x14\n\x05valid\x18\x02 \x01(\x08R\x05valid\x12\x14\n\x05\x65rror\x18\x03 \x01(\x0cR\x05\x65rror\"\x96\x01\n\nGetRequest\x12\x10\n\x03ids\x18\x04 \x03(\x0cR\x03ids\x12\x18\n\x07timeout\x18\x02 \x01(\x02R\x07timeout\x12\"\n\x0c\x61synchronous\x18\x03 \x01(\x08R\x0c\x61synchronous\x12$\n\x0estart_chunk_id\x18\x05 \x01(\x05R\x0cstartChunkId\x12\x12\n\x02id\x18\x01 \x01(\x0c\x42\x02\x18\x01R\x02id\"\xaa\x01\n\x0bGetResponse\x12\x14\n\x05valid\x18\x01 \x01(\x08R\x05valid\x12\x12\n\x04\x64\x61ta\x18\x02 \x01(\x0cR\x04\x64\x61ta\x12\x14\n\x05\x65rror\x18\x03 \x01(\x0cR\x05\x65rror\x12\x19\n\x08\x63hunk_id\x18\x04 \x01(\x05R\x07\x63hunkId\x12!\n\x0ctotal_chunks\x18\x05 \x01(\x05R\x0btotalChunks\x12\x1d\n\ntotal_size\x18\x06 \x01(\x04R\ttotalSize\"\x84\x01\n\x0bWaitRequest\x12\x1d\n\nobject_ids\x18\x01 \x03(\x0cR\tobjectIds\x12\x1f\n\x0bnum_returns\x18\x02 \x01(\x03R\nnumReturns\x12\x18\n\x07timeout\x18\x03 \x01(\x01R\x07timeout\x12\x1b\n\tclient_id\x18\x04 \x01(\tR\x08\x63lientId\"\x80\x01\n\x0cWaitResponse\x12\x14\n\x05valid\x18\x01 \x01(\x08R\x05valid\x12(\n\x10ready_object_ids\x18\x02 \x03(\x0cR\x0ereadyObjectIds\x12\x30\n\x14remaining_object_ids\x18\x03 \x03(\x0cR\x12remainingObjectIds\"\xad\x01\n\x0f\x43lusterInfoType\"\x99\x01\n\x08TypeEnum\x12\x12\n\x0eIS_INITIALIZED\x10\x00\x12\t\n\x05NODES\x10\x01\x12\x15\n\x11\x43LUSTER_RESOURCES\x10\x02\x12\x17\n\x13\x41VAILABLE_RESOURCES\x10\x03\x12\x13\n\x0fRUNTIME_CONTEXT\x10\x04\x12\x0c\n\x08TIMELINE\x10\x05\x12\x08\n\x04PING\x10\x06\x12\x11\n\rDASHBOARD_URL\x10\x07\"K\n\x12\x43lusterInfoRequest\x12\x35\n\x04type\x18\x01 \x01(\x0e\x32!.ray.rpc.ClusterInfoType.TypeEnumR\x04type\"\x8e\x05\n\x13\x43lusterInfoResponse\x12\x35\n\x04type\x18\x01 \x01(\x0e\x32!.ray.rpc.ClusterInfoType.TypeEnumR\x04type\x12\x14\n\x04json\x18\x02 \x01(\tH\x00R\x04json\x12S\n\x0eresource_table\x18\x03 \x01(\x0b\x32*.ray.rpc.ClusterInfoResponse.ResourceTableH\x00R\rresourceTable\x12V\n\x0fruntime_context\x18\x04 \x01(\x0b\x32+.ray.rpc.ClusterInfoResponse.RuntimeContextH\x00R\x0eruntimeContext\x1a\x96\x01\n\rResourceTable\x12K\n\x05table\x18\x01 \x03(\x0b\x32\x35.ray.rpc.ClusterInfoResponse.ResourceTable.TableEntryR\x05table\x1a\x38\n\nTableEntry\x12\x10\n\x03key\x18\x01 \x01(\tR\x03key\x12\x14\n\x05value\x18\x02 \x01(\x01R\x05value:\x02\x38\x01\x1a\xd2\x01\n\x0eRuntimeContext\x12\x15\n\x06job_id\x18\x01 \x01(\x0cR\x05jobId\x12\x17\n\x07node_id\x18\x02 \x01(\x0cR\x06nodeId\x12\x1c\n\tnamespace\x18\x03 \x01(\tR\tnamespace\x12\x1f\n\x0bruntime_env\x18\x04 \x01(\tR\nruntimeEnv\x12\x30\n\x14\x63\x61pture_client_tasks\x18\x05 \x01(\x08R\x12\x63\x61ptureClientTasks\x12\x1f\n\x0bgcs_address\x18\x06 \x01(\tR\ngcsAddressB\x0f\n\rresponse_type\"\xf1\x02\n\x10TerminateRequest\x12\x1b\n\tclient_id\x18\x01 \x01(\tR\x08\x63lientId\x12@\n\x05\x61\x63tor\x18\x02 \x01(\x0b\x32(.ray.rpc.TerminateRequest.ActorTerminateH\x00R\x05\x61\x63tor\x12P\n\x0btask_object\x18\x03 \x01(\x0b\x32-.ray.rpc.TerminateRequest.TaskObjectTerminateH\x00R\ntaskObject\x1a?\n\x0e\x41\x63torTerminate\x12\x0e\n\x02id\x18\x01 \x01(\x0cR\x02id\x12\x1d\n\nno_restart\x18\x02 \x01(\x08R\tnoRestart\x1aY\n\x13TaskObjectTerminate\x12\x0e\n\x02id\x18\x01 \x01(\x0cR\x02id\x12\x14\n\x05\x66orce\x18\x02 \x01(\x08R\x05\x66orce\x12\x1c\n\trecursive\x18\x03 \x01(\x08R\trecursiveB\x10\n\x0eterminate_type\"#\n\x11TerminateResponse\x12\x0e\n\x02ok\x18\x01 \x01(\x08R\x02ok\"T\n\x0fKVExistsRequest\x12\x10\n\x03key\x18\x01 \x01(\x0cR\x03key\x12!\n\tnamespace\x18\x02 \x01(\x0cH\x00R\tnamespace\x88\x01\x01\x42\x0c\n\n_namespace\"*\n\x10KVExistsResponse\x12\x16\n\x06\x65xists\x18\x01 \x01(\x08R\x06\x65xists\"Q\n\x0cKVGetRequest\x12\x10\n\x03key\x18\x01 \x01(\x0cR\x03key\x12!\n\tnamespace\x18\x02 \x01(\x0cH\x00R\tnamespace\x88\x01\x01\x42\x0c\n\n_namespace\"4\n\rKVGetResponse\x12\x19\n\x05value\x18\x01 \x01(\x0cH\x00R\x05value\x88\x01\x01\x42\x08\n\x06_value\"\x85\x01\n\x0cKVPutRequest\x12\x10\n\x03key\x18\x01 \x01(\x0cR\x03key\x12\x14\n\x05value\x18\x02 \x01(\x0cR\x05value\x12\x1c\n\toverwrite\x18\x03 \x01(\x08R\toverwrite\x12!\n\tnamespace\x18\x04 \x01(\x0cH\x00R\tnamespace\x88\x01\x01\x42\x0c\n\n_namespace\"6\n\rKVPutResponse\x12%\n\x0e\x61lready_exists\x18\x01 \x01(\x08R\ralreadyExists\"u\n\x0cKVDelRequest\x12\x10\n\x03key\x18\x01 \x01(\x0cR\x03key\x12\"\n\rdel_by_prefix\x18\x02 \x01(\x08R\x0b\x64\x65lByPrefix\x12!\n\tnamespace\x18\x03 \x01(\x0cH\x00R\tnamespace\x88\x01\x01\x42\x0c\n\n_namespace\"0\n\rKVDelResponse\x12\x1f\n\x0b\x64\x65leted_num\x18\x01 \x01(\x05R\ndeletedNum\"X\n\rKVListRequest\x12\x16\n\x06prefix\x18\x01 \x01(\x0cR\x06prefix\x12!\n\tnamespace\x18\x02 \x01(\x0cH\x00R\tnamespace\x88\x01\x01\x42\x0c\n\n_namespace\"$\n\x0eKVListResponse\x12\x12\n\x04keys\x18\x01 \x03(\x0cR\x04keys\"T\n\x1d\x43lientPinRuntimeEnvURIRequest\x12\x10\n\x03uri\x18\x01 \x01(\tR\x03uri\x12!\n\x0c\x65xpiration_s\x18\x02 \x01(\x05R\x0b\x65xpirationS\" \n\x1e\x43lientPinRuntimeEnvURIResponse\"\x8a\x01\n\x0bInitRequest\x12\x1d\n\njob_config\x18\x01 \x01(\x0cR\tjobConfig\x12&\n\x0fray_init_kwargs\x18\x02 \x01(\tR\rrayInitKwargs\x12\x34\n\x16reconnect_grace_period\x18\x03 \x01(\x05R\x14reconnectGracePeriod\"0\n\x0cInitResponse\x12\x0e\n\x02ok\x18\x01 \x01(\x08R\x02ok\x12\x10\n\x03msg\x18\x02 \x01(\tR\x03msg\"\x17\n\x15PrepRuntimeEnvRequest\"\x18\n\x16PrepRuntimeEnvResponse\"E\n\x1c\x43lientListNamedActorsRequest\x12%\n\x0e\x61ll_namespaces\x18\x01 \x01(\x08R\rallNamespaces\"@\n\x1d\x43lientListNamedActorsResponse\x12\x1f\n\x0b\x61\x63tors_json\x18\x01 \x01(\tR\nactorsJson\"\"\n\x0eReleaseRequest\x12\x10\n\x03ids\x18\x01 \x03(\x0cR\x03ids\"!\n\x0fReleaseResponse\x12\x0e\n\x02ok\x18\x02 \x03(\x08R\x02ok\"\x17\n\x15\x43onnectionInfoRequest\"\xcb\x01\n\x16\x43onnectionInfoResponse\x12\x1f\n\x0bnum_clients\x18\x01 \x01(\x05R\nnumClients\x12\x1f\n\x0bray_version\x18\x02 \x01(\tR\nrayVersion\x12\x1d\n\nray_commit\x18\x03 \x01(\tR\trayCommit\x12%\n\x0epython_version\x18\x04 \x01(\tR\rpythonVersion\x12)\n\x10protocol_version\x18\x05 \x01(\tR\x0fprotocolVersion\"\x1a\n\x18\x43onnectionCleanupRequest\"\x1b\n\x19\x43onnectionCleanupResponse\"+\n\x12\x41\x63knowledgeRequest\x12\x15\n\x06req_id\x18\x01 \x01(\x05R\x05reqId\"\xc6\x05\n\x0b\x44\x61taRequest\x12\x15\n\x06req_id\x18\x01 \x01(\x05R\x05reqId\x12\'\n\x03get\x18\x02 \x01(\x0b\x32\x13.ray.rpc.GetRequestH\x00R\x03get\x12\'\n\x03put\x18\x03 \x01(\x0b\x32\x13.ray.rpc.PutRequestH\x00R\x03put\x12\x33\n\x07release\x18\x04 \x01(\x0b\x32\x17.ray.rpc.ReleaseRequestH\x00R\x07release\x12I\n\x0f\x63onnection_info\x18\x05 \x01(\x0b\x32\x1e.ray.rpc.ConnectionInfoRequestH\x00R\x0e\x63onnectionInfo\x12*\n\x04init\x18\x06 \x01(\x0b\x32\x14.ray.rpc.InitRequestH\x00R\x04init\x12J\n\x10prep_runtime_env\x18\x07 \x01(\x0b\x32\x1e.ray.rpc.PrepRuntimeEnvRequestH\x00R\x0eprepRuntimeEnv\x12R\n\x12\x63onnection_cleanup\x18\x08 \x01(\x0b\x32!.ray.rpc.ConnectionCleanupRequestH\x00R\x11\x63onnectionCleanup\x12?\n\x0b\x61\x63knowledge\x18\t \x01(\x0b\x32\x1b.ray.rpc.AcknowledgeRequestH\x00R\x0b\x61\x63knowledge\x12)\n\x04task\x18\n \x01(\x0b\x32\x13.ray.rpc.ClientTaskH\x00R\x04task\x12\x39\n\tterminate\x18\x0b \x01(\x0b\x32\x19.ray.rpc.TerminateRequestH\x00R\tterminate\x12S\n\x11list_named_actors\x18\x0c \x01(\x0b\x32%.ray.rpc.ClientListNamedActorsRequestH\x00R\x0flistNamedActorsB\x06\n\x04type\"\xb5\x05\n\x0c\x44\x61taResponse\x12\x15\n\x06req_id\x18\x01 \x01(\x05R\x05reqId\x12(\n\x03get\x18\x02 \x01(\x0b\x32\x14.ray.rpc.GetResponseH\x00R\x03get\x12(\n\x03put\x18\x03 \x01(\x0b\x32\x14.ray.rpc.PutResponseH\x00R\x03put\x12\x34\n\x07release\x18\x04 \x01(\x0b\x32\x18.ray.rpc.ReleaseResponseH\x00R\x07release\x12J\n\x0f\x63onnection_info\x18\x05 \x01(\x0b\x32\x1f.ray.rpc.ConnectionInfoResponseH\x00R\x0e\x63onnectionInfo\x12+\n\x04init\x18\x06 \x01(\x0b\x32\x15.ray.rpc.InitResponseH\x00R\x04init\x12K\n\x10prep_runtime_env\x18\x07 \x01(\x0b\x32\x1f.ray.rpc.PrepRuntimeEnvResponseH\x00R\x0eprepRuntimeEnv\x12S\n\x12\x63onnection_cleanup\x18\x08 \x01(\x0b\x32\".ray.rpc.ConnectionCleanupResponseH\x00R\x11\x63onnectionCleanup\x12<\n\x0btask_ticket\x18\n \x01(\x0b\x32\x19.ray.rpc.ClientTaskTicketH\x00R\ntaskTicket\x12:\n\tterminate\x18\x0b \x01(\x0b\x32\x1a.ray.rpc.TerminateResponseH\x00R\tterminate\x12T\n\x11list_named_actors\x18\x0c \x01(\x0b\x32&.ray.rpc.ClientListNamedActorsResponseH\x00R\x0flistNamedActorsB\x06\n\x04typeJ\x04\x08\t\x10\nR\x0b\x61\x63knowledge\"J\n\x12LogSettingsRequest\x12\x18\n\x07\x65nabled\x18\x01 \x01(\x08R\x07\x65nabled\x12\x1a\n\x08loglevel\x18\x02 \x01(\x05R\x08loglevel\"E\n\x07LogData\x12\x10\n\x03msg\x18\x01 \x01(\tR\x03msg\x12\x14\n\x05level\x18\x02 \x01(\x05R\x05level\x12\x12\n\x04name\x18\x03 \x01(\tR\x04name*\x13\n\x04Type\x12\x0b\n\x07\x44\x45\x46\x41ULT\x10\x00\x32\x96\x08\n\x0cRayletDriver\x12\x35\n\x04Init\x12\x14.ray.rpc.InitRequest\x1a\x15.ray.rpc.InitResponse\"\x00\x12S\n\x0ePrepRuntimeEnv\x12\x1e.ray.rpc.PrepRuntimeEnvRequest\x1a\x1f.ray.rpc.PrepRuntimeEnvResponse\"\x00\x12:\n\tGetObject\x12\x13.ray.rpc.GetRequest\x1a\x14.ray.rpc.GetResponse\"\x00\x30\x01\x12\x38\n\tPutObject\x12\x13.ray.rpc.PutRequest\x1a\x14.ray.rpc.PutResponse\"\x00\x12;\n\nWaitObject\x12\x14.ray.rpc.WaitRequest\x1a\x15.ray.rpc.WaitResponse\"\x00\x12<\n\x08Schedule\x12\x13.ray.rpc.ClientTask\x1a\x19.ray.rpc.ClientTaskTicket\"\x00\x12\x44\n\tTerminate\x12\x19.ray.rpc.TerminateRequest\x1a\x1a.ray.rpc.TerminateResponse\"\x00\x12J\n\x0b\x43lusterInfo\x12\x1b.ray.rpc.ClusterInfoRequest\x1a\x1c.ray.rpc.ClusterInfoResponse\"\x00\x12\x38\n\x05KVGet\x12\x15.ray.rpc.KVGetRequest\x1a\x16.ray.rpc.KVGetResponse\"\x00\x12\x38\n\x05KVPut\x12\x15.ray.rpc.KVPutRequest\x1a\x16.ray.rpc.KVPutResponse\"\x00\x12\x38\n\x05KVDel\x12\x15.ray.rpc.KVDelRequest\x1a\x16.ray.rpc.KVDelResponse\"\x00\x12;\n\x06KVList\x12\x16.ray.rpc.KVListRequest\x1a\x17.ray.rpc.KVListResponse\"\x00\x12\x41\n\x08KVExists\x12\x18.ray.rpc.KVExistsRequest\x1a\x19.ray.rpc.KVExistsResponse\"\x00\x12\x62\n\x0fListNamedActors\x12%.ray.rpc.ClientListNamedActorsRequest\x1a&.ray.rpc.ClientListNamedActorsResponse\"\x00\x12\x65\n\x10PinRuntimeEnvURI\x12&.ray.rpc.ClientPinRuntimeEnvURIRequest\x1a\'.ray.rpc.ClientPinRuntimeEnvURIResponse\"\x00\x32S\n\x12RayletDataStreamer\x12=\n\x08\x44\x61tapath\x12\x14.ray.rpc.DataRequest\x1a\x15.ray.rpc.DataResponse\"\x00(\x01\x30\x01\x32U\n\x11RayletLogStreamer\x12@\n\tLogstream\x12\x1b.ray.rpc.LogSettingsRequest\x1a\x10.ray.rpc.LogData\"\x00(\x01\x30\x01\x42\x03\xf8\x01\x01\x62\x06proto3')

_TYPE = DESCRIPTOR.enum_types_by_name['Type']
Type = enum_type_wrapper.EnumTypeWrapper(_TYPE)
DEFAULT = 0


_ARG = DESCRIPTOR.message_types_by_name['Arg']
_TASKOPTIONS = DESCRIPTOR.message_types_by_name['TaskOptions']
_CLIENTTASK = DESCRIPTOR.message_types_by_name['ClientTask']
_CLIENTTASK_KWARGSENTRY = _CLIENTTASK.nested_types_by_name['KwargsEntry']
_CLIENTTASKTICKET = DESCRIPTOR.message_types_by_name['ClientTaskTicket']
_PUTREQUEST = DESCRIPTOR.message_types_by_name['PutRequest']
_PUTRESPONSE = DESCRIPTOR.message_types_by_name['PutResponse']
_GETREQUEST = DESCRIPTOR.message_types_by_name['GetRequest']
_GETRESPONSE = DESCRIPTOR.message_types_by_name['GetResponse']
_WAITREQUEST = DESCRIPTOR.message_types_by_name['WaitRequest']
_WAITRESPONSE = DESCRIPTOR.message_types_by_name['WaitResponse']
_CLUSTERINFOTYPE = DESCRIPTOR.message_types_by_name['ClusterInfoType']
_CLUSTERINFOREQUEST = DESCRIPTOR.message_types_by_name['ClusterInfoRequest']
_CLUSTERINFORESPONSE = DESCRIPTOR.message_types_by_name['ClusterInfoResponse']
_CLUSTERINFORESPONSE_RESOURCETABLE = _CLUSTERINFORESPONSE.nested_types_by_name['ResourceTable']
_CLUSTERINFORESPONSE_RESOURCETABLE_TABLEENTRY = _CLUSTERINFORESPONSE_RESOURCETABLE.nested_types_by_name['TableEntry']
_CLUSTERINFORESPONSE_RUNTIMECONTEXT = _CLUSTERINFORESPONSE.nested_types_by_name['RuntimeContext']
_TERMINATEREQUEST = DESCRIPTOR.message_types_by_name['TerminateRequest']
_TERMINATEREQUEST_ACTORTERMINATE = _TERMINATEREQUEST.nested_types_by_name['ActorTerminate']
_TERMINATEREQUEST_TASKOBJECTTERMINATE = _TERMINATEREQUEST.nested_types_by_name['TaskObjectTerminate']
_TERMINATERESPONSE = DESCRIPTOR.message_types_by_name['TerminateResponse']
_KVEXISTSREQUEST = DESCRIPTOR.message_types_by_name['KVExistsRequest']
_KVEXISTSRESPONSE = DESCRIPTOR.message_types_by_name['KVExistsResponse']
_KVGETREQUEST = DESCRIPTOR.message_types_by_name['KVGetRequest']
_KVGETRESPONSE = DESCRIPTOR.message_types_by_name['KVGetResponse']
_KVPUTREQUEST = DESCRIPTOR.message_types_by_name['KVPutRequest']
_KVPUTRESPONSE = DESCRIPTOR.message_types_by_name['KVPutResponse']
_KVDELREQUEST = DESCRIPTOR.message_types_by_name['KVDelRequest']
_KVDELRESPONSE = DESCRIPTOR.message_types_by_name['KVDelResponse']
_KVLISTREQUEST = DESCRIPTOR.message_types_by_name['KVListRequest']
_KVLISTRESPONSE = DESCRIPTOR.message_types_by_name['KVListResponse']
_CLIENTPINRUNTIMEENVURIREQUEST = DESCRIPTOR.message_types_by_name['ClientPinRuntimeEnvURIRequest']
_CLIENTPINRUNTIMEENVURIRESPONSE = DESCRIPTOR.message_types_by_name['ClientPinRuntimeEnvURIResponse']
_INITREQUEST = DESCRIPTOR.message_types_by_name['InitRequest']
_INITRESPONSE = DESCRIPTOR.message_types_by_name['InitResponse']
_PREPRUNTIMEENVREQUEST = DESCRIPTOR.message_types_by_name['PrepRuntimeEnvRequest']
_PREPRUNTIMEENVRESPONSE = DESCRIPTOR.message_types_by_name['PrepRuntimeEnvResponse']
_CLIENTLISTNAMEDACTORSREQUEST = DESCRIPTOR.message_types_by_name['ClientListNamedActorsRequest']
_CLIENTLISTNAMEDACTORSRESPONSE = DESCRIPTOR.message_types_by_name['ClientListNamedActorsResponse']
_RELEASEREQUEST = DESCRIPTOR.message_types_by_name['ReleaseRequest']
_RELEASERESPONSE = DESCRIPTOR.message_types_by_name['ReleaseResponse']
_CONNECTIONINFOREQUEST = DESCRIPTOR.message_types_by_name['ConnectionInfoRequest']
_CONNECTIONINFORESPONSE = DESCRIPTOR.message_types_by_name['ConnectionInfoResponse']
_CONNECTIONCLEANUPREQUEST = DESCRIPTOR.message_types_by_name['ConnectionCleanupRequest']
_CONNECTIONCLEANUPRESPONSE = DESCRIPTOR.message_types_by_name['ConnectionCleanupResponse']
_ACKNOWLEDGEREQUEST = DESCRIPTOR.message_types_by_name['AcknowledgeRequest']
_DATAREQUEST = DESCRIPTOR.message_types_by_name['DataRequest']
_DATARESPONSE = DESCRIPTOR.message_types_by_name['DataResponse']
_LOGSETTINGSREQUEST = DESCRIPTOR.message_types_by_name['LogSettingsRequest']
_LOGDATA = DESCRIPTOR.message_types_by_name['LogData']
_ARG_LOCALITY = _ARG.enum_types_by_name['Locality']
_CLIENTTASK_REMOTEEXECTYPE = _CLIENTTASK.enum_types_by_name['RemoteExecType']
_CLUSTERINFOTYPE_TYPEENUM = _CLUSTERINFOTYPE.enum_types_by_name['TypeEnum']
Arg = _reflection.GeneratedProtocolMessageType('Arg', (_message.Message,), {
  'DESCRIPTOR' : _ARG,
  '__module__' : 'src.ray.protobuf.ray_client_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.Arg)
  })
_sym_db.RegisterMessage(Arg)

TaskOptions = _reflection.GeneratedProtocolMessageType('TaskOptions', (_message.Message,), {
  'DESCRIPTOR' : _TASKOPTIONS,
  '__module__' : 'src.ray.protobuf.ray_client_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.TaskOptions)
  })
_sym_db.RegisterMessage(TaskOptions)

ClientTask = _reflection.GeneratedProtocolMessageType('ClientTask', (_message.Message,), {

  'KwargsEntry' : _reflection.GeneratedProtocolMessageType('KwargsEntry', (_message.Message,), {
    'DESCRIPTOR' : _CLIENTTASK_KWARGSENTRY,
    '__module__' : 'src.ray.protobuf.ray_client_pb2'
    # @@protoc_insertion_point(class_scope:ray.rpc.ClientTask.KwargsEntry)
    })
  ,
  'DESCRIPTOR' : _CLIENTTASK,
  '__module__' : 'src.ray.protobuf.ray_client_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.ClientTask)
  })
_sym_db.RegisterMessage(ClientTask)
_sym_db.RegisterMessage(ClientTask.KwargsEntry)

ClientTaskTicket = _reflection.GeneratedProtocolMessageType('ClientTaskTicket', (_message.Message,), {
  'DESCRIPTOR' : _CLIENTTASKTICKET,
  '__module__' : 'src.ray.protobuf.ray_client_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.ClientTaskTicket)
  })
_sym_db.RegisterMessage(ClientTaskTicket)

PutRequest = _reflection.GeneratedProtocolMessageType('PutRequest', (_message.Message,), {
  'DESCRIPTOR' : _PUTREQUEST,
  '__module__' : 'src.ray.protobuf.ray_client_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.PutRequest)
  })
_sym_db.RegisterMessage(PutRequest)

PutResponse = _reflection.GeneratedProtocolMessageType('PutResponse', (_message.Message,), {
  'DESCRIPTOR' : _PUTRESPONSE,
  '__module__' : 'src.ray.protobuf.ray_client_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.PutResponse)
  })
_sym_db.RegisterMessage(PutResponse)

GetRequest = _reflection.GeneratedProtocolMessageType('GetRequest', (_message.Message,), {
  'DESCRIPTOR' : _GETREQUEST,
  '__module__' : 'src.ray.protobuf.ray_client_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.GetRequest)
  })
_sym_db.RegisterMessage(GetRequest)

GetResponse = _reflection.GeneratedProtocolMessageType('GetResponse', (_message.Message,), {
  'DESCRIPTOR' : _GETRESPONSE,
  '__module__' : 'src.ray.protobuf.ray_client_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.GetResponse)
  })
_sym_db.RegisterMessage(GetResponse)

WaitRequest = _reflection.GeneratedProtocolMessageType('WaitRequest', (_message.Message,), {
  'DESCRIPTOR' : _WAITREQUEST,
  '__module__' : 'src.ray.protobuf.ray_client_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.WaitRequest)
  })
_sym_db.RegisterMessage(WaitRequest)

WaitResponse = _reflection.GeneratedProtocolMessageType('WaitResponse', (_message.Message,), {
  'DESCRIPTOR' : _WAITRESPONSE,
  '__module__' : 'src.ray.protobuf.ray_client_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.WaitResponse)
  })
_sym_db.RegisterMessage(WaitResponse)

ClusterInfoType = _reflection.GeneratedProtocolMessageType('ClusterInfoType', (_message.Message,), {
  'DESCRIPTOR' : _CLUSTERINFOTYPE,
  '__module__' : 'src.ray.protobuf.ray_client_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.ClusterInfoType)
  })
_sym_db.RegisterMessage(ClusterInfoType)

ClusterInfoRequest = _reflection.GeneratedProtocolMessageType('ClusterInfoRequest', (_message.Message,), {
  'DESCRIPTOR' : _CLUSTERINFOREQUEST,
  '__module__' : 'src.ray.protobuf.ray_client_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.ClusterInfoRequest)
  })
_sym_db.RegisterMessage(ClusterInfoRequest)

ClusterInfoResponse = _reflection.GeneratedProtocolMessageType('ClusterInfoResponse', (_message.Message,), {

  'ResourceTable' : _reflection.GeneratedProtocolMessageType('ResourceTable', (_message.Message,), {

    'TableEntry' : _reflection.GeneratedProtocolMessageType('TableEntry', (_message.Message,), {
      'DESCRIPTOR' : _CLUSTERINFORESPONSE_RESOURCETABLE_TABLEENTRY,
      '__module__' : 'src.ray.protobuf.ray_client_pb2'
      # @@protoc_insertion_point(class_scope:ray.rpc.ClusterInfoResponse.ResourceTable.TableEntry)
      })
    ,
    'DESCRIPTOR' : _CLUSTERINFORESPONSE_RESOURCETABLE,
    '__module__' : 'src.ray.protobuf.ray_client_pb2'
    # @@protoc_insertion_point(class_scope:ray.rpc.ClusterInfoResponse.ResourceTable)
    })
  ,

  'RuntimeContext' : _reflection.GeneratedProtocolMessageType('RuntimeContext', (_message.Message,), {
    'DESCRIPTOR' : _CLUSTERINFORESPONSE_RUNTIMECONTEXT,
    '__module__' : 'src.ray.protobuf.ray_client_pb2'
    # @@protoc_insertion_point(class_scope:ray.rpc.ClusterInfoResponse.RuntimeContext)
    })
  ,
  'DESCRIPTOR' : _CLUSTERINFORESPONSE,
  '__module__' : 'src.ray.protobuf.ray_client_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.ClusterInfoResponse)
  })
_sym_db.RegisterMessage(ClusterInfoResponse)
_sym_db.RegisterMessage(ClusterInfoResponse.ResourceTable)
_sym_db.RegisterMessage(ClusterInfoResponse.ResourceTable.TableEntry)
_sym_db.RegisterMessage(ClusterInfoResponse.RuntimeContext)

TerminateRequest = _reflection.GeneratedProtocolMessageType('TerminateRequest', (_message.Message,), {

  'ActorTerminate' : _reflection.GeneratedProtocolMessageType('ActorTerminate', (_message.Message,), {
    'DESCRIPTOR' : _TERMINATEREQUEST_ACTORTERMINATE,
    '__module__' : 'src.ray.protobuf.ray_client_pb2'
    # @@protoc_insertion_point(class_scope:ray.rpc.TerminateRequest.ActorTerminate)
    })
  ,

  'TaskObjectTerminate' : _reflection.GeneratedProtocolMessageType('TaskObjectTerminate', (_message.Message,), {
    'DESCRIPTOR' : _TERMINATEREQUEST_TASKOBJECTTERMINATE,
    '__module__' : 'src.ray.protobuf.ray_client_pb2'
    # @@protoc_insertion_point(class_scope:ray.rpc.TerminateRequest.TaskObjectTerminate)
    })
  ,
  'DESCRIPTOR' : _TERMINATEREQUEST,
  '__module__' : 'src.ray.protobuf.ray_client_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.TerminateRequest)
  })
_sym_db.RegisterMessage(TerminateRequest)
_sym_db.RegisterMessage(TerminateRequest.ActorTerminate)
_sym_db.RegisterMessage(TerminateRequest.TaskObjectTerminate)

TerminateResponse = _reflection.GeneratedProtocolMessageType('TerminateResponse', (_message.Message,), {
  'DESCRIPTOR' : _TERMINATERESPONSE,
  '__module__' : 'src.ray.protobuf.ray_client_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.TerminateResponse)
  })
_sym_db.RegisterMessage(TerminateResponse)

KVExistsRequest = _reflection.GeneratedProtocolMessageType('KVExistsRequest', (_message.Message,), {
  'DESCRIPTOR' : _KVEXISTSREQUEST,
  '__module__' : 'src.ray.protobuf.ray_client_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.KVExistsRequest)
  })
_sym_db.RegisterMessage(KVExistsRequest)

KVExistsResponse = _reflection.GeneratedProtocolMessageType('KVExistsResponse', (_message.Message,), {
  'DESCRIPTOR' : _KVEXISTSRESPONSE,
  '__module__' : 'src.ray.protobuf.ray_client_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.KVExistsResponse)
  })
_sym_db.RegisterMessage(KVExistsResponse)

KVGetRequest = _reflection.GeneratedProtocolMessageType('KVGetRequest', (_message.Message,), {
  'DESCRIPTOR' : _KVGETREQUEST,
  '__module__' : 'src.ray.protobuf.ray_client_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.KVGetRequest)
  })
_sym_db.RegisterMessage(KVGetRequest)

KVGetResponse = _reflection.GeneratedProtocolMessageType('KVGetResponse', (_message.Message,), {
  'DESCRIPTOR' : _KVGETRESPONSE,
  '__module__' : 'src.ray.protobuf.ray_client_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.KVGetResponse)
  })
_sym_db.RegisterMessage(KVGetResponse)

KVPutRequest = _reflection.GeneratedProtocolMessageType('KVPutRequest', (_message.Message,), {
  'DESCRIPTOR' : _KVPUTREQUEST,
  '__module__' : 'src.ray.protobuf.ray_client_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.KVPutRequest)
  })
_sym_db.RegisterMessage(KVPutRequest)

KVPutResponse = _reflection.GeneratedProtocolMessageType('KVPutResponse', (_message.Message,), {
  'DESCRIPTOR' : _KVPUTRESPONSE,
  '__module__' : 'src.ray.protobuf.ray_client_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.KVPutResponse)
  })
_sym_db.RegisterMessage(KVPutResponse)

KVDelRequest = _reflection.GeneratedProtocolMessageType('KVDelRequest', (_message.Message,), {
  'DESCRIPTOR' : _KVDELREQUEST,
  '__module__' : 'src.ray.protobuf.ray_client_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.KVDelRequest)
  })
_sym_db.RegisterMessage(KVDelRequest)

KVDelResponse = _reflection.GeneratedProtocolMessageType('KVDelResponse', (_message.Message,), {
  'DESCRIPTOR' : _KVDELRESPONSE,
  '__module__' : 'src.ray.protobuf.ray_client_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.KVDelResponse)
  })
_sym_db.RegisterMessage(KVDelResponse)

KVListRequest = _reflection.GeneratedProtocolMessageType('KVListRequest', (_message.Message,), {
  'DESCRIPTOR' : _KVLISTREQUEST,
  '__module__' : 'src.ray.protobuf.ray_client_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.KVListRequest)
  })
_sym_db.RegisterMessage(KVListRequest)

KVListResponse = _reflection.GeneratedProtocolMessageType('KVListResponse', (_message.Message,), {
  'DESCRIPTOR' : _KVLISTRESPONSE,
  '__module__' : 'src.ray.protobuf.ray_client_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.KVListResponse)
  })
_sym_db.RegisterMessage(KVListResponse)

ClientPinRuntimeEnvURIRequest = _reflection.GeneratedProtocolMessageType('ClientPinRuntimeEnvURIRequest', (_message.Message,), {
  'DESCRIPTOR' : _CLIENTPINRUNTIMEENVURIREQUEST,
  '__module__' : 'src.ray.protobuf.ray_client_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.ClientPinRuntimeEnvURIRequest)
  })
_sym_db.RegisterMessage(ClientPinRuntimeEnvURIRequest)

ClientPinRuntimeEnvURIResponse = _reflection.GeneratedProtocolMessageType('ClientPinRuntimeEnvURIResponse', (_message.Message,), {
  'DESCRIPTOR' : _CLIENTPINRUNTIMEENVURIRESPONSE,
  '__module__' : 'src.ray.protobuf.ray_client_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.ClientPinRuntimeEnvURIResponse)
  })
_sym_db.RegisterMessage(ClientPinRuntimeEnvURIResponse)

InitRequest = _reflection.GeneratedProtocolMessageType('InitRequest', (_message.Message,), {
  'DESCRIPTOR' : _INITREQUEST,
  '__module__' : 'src.ray.protobuf.ray_client_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.InitRequest)
  })
_sym_db.RegisterMessage(InitRequest)

InitResponse = _reflection.GeneratedProtocolMessageType('InitResponse', (_message.Message,), {
  'DESCRIPTOR' : _INITRESPONSE,
  '__module__' : 'src.ray.protobuf.ray_client_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.InitResponse)
  })
_sym_db.RegisterMessage(InitResponse)

PrepRuntimeEnvRequest = _reflection.GeneratedProtocolMessageType('PrepRuntimeEnvRequest', (_message.Message,), {
  'DESCRIPTOR' : _PREPRUNTIMEENVREQUEST,
  '__module__' : 'src.ray.protobuf.ray_client_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.PrepRuntimeEnvRequest)
  })
_sym_db.RegisterMessage(PrepRuntimeEnvRequest)

PrepRuntimeEnvResponse = _reflection.GeneratedProtocolMessageType('PrepRuntimeEnvResponse', (_message.Message,), {
  'DESCRIPTOR' : _PREPRUNTIMEENVRESPONSE,
  '__module__' : 'src.ray.protobuf.ray_client_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.PrepRuntimeEnvResponse)
  })
_sym_db.RegisterMessage(PrepRuntimeEnvResponse)

ClientListNamedActorsRequest = _reflection.GeneratedProtocolMessageType('ClientListNamedActorsRequest', (_message.Message,), {
  'DESCRIPTOR' : _CLIENTLISTNAMEDACTORSREQUEST,
  '__module__' : 'src.ray.protobuf.ray_client_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.ClientListNamedActorsRequest)
  })
_sym_db.RegisterMessage(ClientListNamedActorsRequest)

ClientListNamedActorsResponse = _reflection.GeneratedProtocolMessageType('ClientListNamedActorsResponse', (_message.Message,), {
  'DESCRIPTOR' : _CLIENTLISTNAMEDACTORSRESPONSE,
  '__module__' : 'src.ray.protobuf.ray_client_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.ClientListNamedActorsResponse)
  })
_sym_db.RegisterMessage(ClientListNamedActorsResponse)

ReleaseRequest = _reflection.GeneratedProtocolMessageType('ReleaseRequest', (_message.Message,), {
  'DESCRIPTOR' : _RELEASEREQUEST,
  '__module__' : 'src.ray.protobuf.ray_client_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.ReleaseRequest)
  })
_sym_db.RegisterMessage(ReleaseRequest)

ReleaseResponse = _reflection.GeneratedProtocolMessageType('ReleaseResponse', (_message.Message,), {
  'DESCRIPTOR' : _RELEASERESPONSE,
  '__module__' : 'src.ray.protobuf.ray_client_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.ReleaseResponse)
  })
_sym_db.RegisterMessage(ReleaseResponse)

ConnectionInfoRequest = _reflection.GeneratedProtocolMessageType('ConnectionInfoRequest', (_message.Message,), {
  'DESCRIPTOR' : _CONNECTIONINFOREQUEST,
  '__module__' : 'src.ray.protobuf.ray_client_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.ConnectionInfoRequest)
  })
_sym_db.RegisterMessage(ConnectionInfoRequest)

ConnectionInfoResponse = _reflection.GeneratedProtocolMessageType('ConnectionInfoResponse', (_message.Message,), {
  'DESCRIPTOR' : _CONNECTIONINFORESPONSE,
  '__module__' : 'src.ray.protobuf.ray_client_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.ConnectionInfoResponse)
  })
_sym_db.RegisterMessage(ConnectionInfoResponse)

ConnectionCleanupRequest = _reflection.GeneratedProtocolMessageType('ConnectionCleanupRequest', (_message.Message,), {
  'DESCRIPTOR' : _CONNECTIONCLEANUPREQUEST,
  '__module__' : 'src.ray.protobuf.ray_client_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.ConnectionCleanupRequest)
  })
_sym_db.RegisterMessage(ConnectionCleanupRequest)

ConnectionCleanupResponse = _reflection.GeneratedProtocolMessageType('ConnectionCleanupResponse', (_message.Message,), {
  'DESCRIPTOR' : _CONNECTIONCLEANUPRESPONSE,
  '__module__' : 'src.ray.protobuf.ray_client_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.ConnectionCleanupResponse)
  })
_sym_db.RegisterMessage(ConnectionCleanupResponse)

AcknowledgeRequest = _reflection.GeneratedProtocolMessageType('AcknowledgeRequest', (_message.Message,), {
  'DESCRIPTOR' : _ACKNOWLEDGEREQUEST,
  '__module__' : 'src.ray.protobuf.ray_client_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.AcknowledgeRequest)
  })
_sym_db.RegisterMessage(AcknowledgeRequest)

DataRequest = _reflection.GeneratedProtocolMessageType('DataRequest', (_message.Message,), {
  'DESCRIPTOR' : _DATAREQUEST,
  '__module__' : 'src.ray.protobuf.ray_client_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.DataRequest)
  })
_sym_db.RegisterMessage(DataRequest)

DataResponse = _reflection.GeneratedProtocolMessageType('DataResponse', (_message.Message,), {
  'DESCRIPTOR' : _DATARESPONSE,
  '__module__' : 'src.ray.protobuf.ray_client_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.DataResponse)
  })
_sym_db.RegisterMessage(DataResponse)

LogSettingsRequest = _reflection.GeneratedProtocolMessageType('LogSettingsRequest', (_message.Message,), {
  'DESCRIPTOR' : _LOGSETTINGSREQUEST,
  '__module__' : 'src.ray.protobuf.ray_client_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.LogSettingsRequest)
  })
_sym_db.RegisterMessage(LogSettingsRequest)

LogData = _reflection.GeneratedProtocolMessageType('LogData', (_message.Message,), {
  'DESCRIPTOR' : _LOGDATA,
  '__module__' : 'src.ray.protobuf.ray_client_pb2'
  # @@protoc_insertion_point(class_scope:ray.rpc.LogData)
  })
_sym_db.RegisterMessage(LogData)

_RAYLETDRIVER = DESCRIPTOR.services_by_name['RayletDriver']
_RAYLETDATASTREAMER = DESCRIPTOR.services_by_name['RayletDataStreamer']
_RAYLETLOGSTREAMER = DESCRIPTOR.services_by_name['RayletLogStreamer']
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'\370\001\001'
  _CLIENTTASK_KWARGSENTRY._options = None
  _CLIENTTASK_KWARGSENTRY._serialized_options = b'8\001'
  _GETREQUEST.fields_by_name['id']._options = None
  _GETREQUEST.fields_by_name['id']._serialized_options = b'\030\001'
  _CLUSTERINFORESPONSE_RESOURCETABLE_TABLEENTRY._options = None
  _CLUSTERINFORESPONSE_RESOURCETABLE_TABLEENTRY._serialized_options = b'8\001'
  _TYPE._serialized_start=6404
  _TYPE._serialized_end=6423
  _ARG._serialized_start=47
  _ARG._serialized_end=228
  _ARG_LOCALITY._serialized_start=189
  _ARG_LOCALITY._serialized_end=228
  _TASKOPTIONS._serialized_start=230
  _TASKOPTIONS._serialized_end=284
  _CLIENTTASK._serialized_start=287
  _CLIENTTASK._serialized_end=915
  _CLIENTTASK_KWARGSENTRY._serialized_start=753
  _CLIENTTASK_KWARGSENTRY._serialized_end=824
  _CLIENTTASK_REMOTEEXECTYPE._serialized_start=826
  _CLIENTTASK_REMOTEEXECTYPE._serialized_end=915
  _CLIENTTASKTICKET._serialized_start=917
  _CLIENTTASKTICKET._serialized_end=1010
  _PUTREQUEST._serialized_start=1013
  _PUTREQUEST._serialized_end=1201
  _PUTRESPONSE._serialized_start=1203
  _PUTRESPONSE._serialized_end=1276
  _GETREQUEST._serialized_start=1279
  _GETREQUEST._serialized_end=1429
  _GETRESPONSE._serialized_start=1432
  _GETRESPONSE._serialized_end=1602
  _WAITREQUEST._serialized_start=1605
  _WAITREQUEST._serialized_end=1737
  _WAITRESPONSE._serialized_start=1740
  _WAITRESPONSE._serialized_end=1868
  _CLUSTERINFOTYPE._serialized_start=1871
  _CLUSTERINFOTYPE._serialized_end=2044
  _CLUSTERINFOTYPE_TYPEENUM._serialized_start=1891
  _CLUSTERINFOTYPE_TYPEENUM._serialized_end=2044
  _CLUSTERINFOREQUEST._serialized_start=2046
  _CLUSTERINFOREQUEST._serialized_end=2121
  _CLUSTERINFORESPONSE._serialized_start=2124
  _CLUSTERINFORESPONSE._serialized_end=2778
  _CLUSTERINFORESPONSE_RESOURCETABLE._serialized_start=2398
  _CLUSTERINFORESPONSE_RESOURCETABLE._serialized_end=2548
  _CLUSTERINFORESPONSE_RESOURCETABLE_TABLEENTRY._serialized_start=2492
  _CLUSTERINFORESPONSE_RESOURCETABLE_TABLEENTRY._serialized_end=2548
  _CLUSTERINFORESPONSE_RUNTIMECONTEXT._serialized_start=2551
  _CLUSTERINFORESPONSE_RUNTIMECONTEXT._serialized_end=2761
  _TERMINATEREQUEST._serialized_start=2781
  _TERMINATEREQUEST._serialized_end=3150
  _TERMINATEREQUEST_ACTORTERMINATE._serialized_start=2978
  _TERMINATEREQUEST_ACTORTERMINATE._serialized_end=3041
  _TERMINATEREQUEST_TASKOBJECTTERMINATE._serialized_start=3043
  _TERMINATEREQUEST_TASKOBJECTTERMINATE._serialized_end=3132
  _TERMINATERESPONSE._serialized_start=3152
  _TERMINATERESPONSE._serialized_end=3187
  _KVEXISTSREQUEST._serialized_start=3189
  _KVEXISTSREQUEST._serialized_end=3273
  _KVEXISTSRESPONSE._serialized_start=3275
  _KVEXISTSRESPONSE._serialized_end=3317
  _KVGETREQUEST._serialized_start=3319
  _KVGETREQUEST._serialized_end=3400
  _KVGETRESPONSE._serialized_start=3402
  _KVGETRESPONSE._serialized_end=3454
  _KVPUTREQUEST._serialized_start=3457
  _KVPUTREQUEST._serialized_end=3590
  _KVPUTRESPONSE._serialized_start=3592
  _KVPUTRESPONSE._serialized_end=3646
  _KVDELREQUEST._serialized_start=3648
  _KVDELREQUEST._serialized_end=3765
  _KVDELRESPONSE._serialized_start=3767
  _KVDELRESPONSE._serialized_end=3815
  _KVLISTREQUEST._serialized_start=3817
  _KVLISTREQUEST._serialized_end=3905
  _KVLISTRESPONSE._serialized_start=3907
  _KVLISTRESPONSE._serialized_end=3943
  _CLIENTPINRUNTIMEENVURIREQUEST._serialized_start=3945
  _CLIENTPINRUNTIMEENVURIREQUEST._serialized_end=4029
  _CLIENTPINRUNTIMEENVURIRESPONSE._serialized_start=4031
  _CLIENTPINRUNTIMEENVURIRESPONSE._serialized_end=4063
  _INITREQUEST._serialized_start=4066
  _INITREQUEST._serialized_end=4204
  _INITRESPONSE._serialized_start=4206
  _INITRESPONSE._serialized_end=4254
  _PREPRUNTIMEENVREQUEST._serialized_start=4256
  _PREPRUNTIMEENVREQUEST._serialized_end=4279
  _PREPRUNTIMEENVRESPONSE._serialized_start=4281
  _PREPRUNTIMEENVRESPONSE._serialized_end=4305
  _CLIENTLISTNAMEDACTORSREQUEST._serialized_start=4307
  _CLIENTLISTNAMEDACTORSREQUEST._serialized_end=4376
  _CLIENTLISTNAMEDACTORSRESPONSE._serialized_start=4378
  _CLIENTLISTNAMEDACTORSRESPONSE._serialized_end=4442
  _RELEASEREQUEST._serialized_start=4444
  _RELEASEREQUEST._serialized_end=4478
  _RELEASERESPONSE._serialized_start=4480
  _RELEASERESPONSE._serialized_end=4513
  _CONNECTIONINFOREQUEST._serialized_start=4515
  _CONNECTIONINFOREQUEST._serialized_end=4538
  _CONNECTIONINFORESPONSE._serialized_start=4541
  _CONNECTIONINFORESPONSE._serialized_end=4744
  _CONNECTIONCLEANUPREQUEST._serialized_start=4746
  _CONNECTIONCLEANUPREQUEST._serialized_end=4772
  _CONNECTIONCLEANUPRESPONSE._serialized_start=4774
  _CONNECTIONCLEANUPRESPONSE._serialized_end=4801
  _ACKNOWLEDGEREQUEST._serialized_start=4803
  _ACKNOWLEDGEREQUEST._serialized_end=4846
  _DATAREQUEST._serialized_start=4849
  _DATAREQUEST._serialized_end=5559
  _DATARESPONSE._serialized_start=5562
  _DATARESPONSE._serialized_end=6255
  _LOGSETTINGSREQUEST._serialized_start=6257
  _LOGSETTINGSREQUEST._serialized_end=6331
  _LOGDATA._serialized_start=6333
  _LOGDATA._serialized_end=6402
  _RAYLETDRIVER._serialized_start=6426
  _RAYLETDRIVER._serialized_end=7472
  _RAYLETDATASTREAMER._serialized_start=7474
  _RAYLETDATASTREAMER._serialized_end=7557
  _RAYLETLOGSTREAMER._serialized_start=7559
  _RAYLETLOGSTREAMER._serialized_end=7644
# @@protoc_insertion_point(module_scope)
