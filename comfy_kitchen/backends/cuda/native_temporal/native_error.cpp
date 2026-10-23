#include "native_error.h"
#include <string>
static thread_local std::string error;
CK_TEMPORAL_EXPORT void ck_temporal_set_error(const char* message) {error=message;}
CK_TEMPORAL_EXPORT const char* ck_temporal_last_error() {return error.c_str();}

CK_TEMPORAL_EXPORT int ck_temporal_abi_version() {return 1;}
CK_TEMPORAL_EXPORT int ck_temporal_arch() {return CK_TEMPORAL_ARCH;}
