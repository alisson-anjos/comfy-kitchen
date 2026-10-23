#include <string>
static thread_local std::string error;
extern "C" void ck_temporal_set_error(const char* message) {error=message;}
extern "C" const char* ck_temporal_last_error() {return error.c_str();}

extern "C" int ck_temporal_abi_version() {return 1;}
extern "C" int ck_temporal_arch() {return CK_TEMPORAL_ARCH;}
