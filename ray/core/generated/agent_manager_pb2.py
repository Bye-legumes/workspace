# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: src/ray/protobuf/agent_manager.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n$src/ray/protobuf/agent_manager.proto\x12\x07ray.rpc*F\n\x0e\x41gentRpcStatus\x12\x17\n\x13\x41GENT_RPC_STATUS_OK\x10\x00\x12\x1b\n\x17\x41GENT_RPC_STATUS_FAILED\x10\x01\x42\x03\xf8\x01\x01\x62\x06proto3')

_AGENTRPCSTATUS = DESCRIPTOR.enum_types_by_name['AgentRpcStatus']
AgentRpcStatus = enum_type_wrapper.EnumTypeWrapper(_AGENTRPCSTATUS)
AGENT_RPC_STATUS_OK = 0
AGENT_RPC_STATUS_FAILED = 1


if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'\370\001\001'
  _AGENTRPCSTATUS._serialized_start=49
  _AGENTRPCSTATUS._serialized_end=119
# @@protoc_insertion_point(module_scope)
