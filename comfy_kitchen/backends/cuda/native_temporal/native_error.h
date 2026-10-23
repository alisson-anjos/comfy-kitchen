#pragma once
#ifdef _WIN32
#define CK_TEMPORAL_EXPORT extern "C" __declspec(dllexport)
#else
#define CK_TEMPORAL_EXPORT extern "C"
#endif
CK_TEMPORAL_EXPORT void ck_temporal_set_error(const char*);
CK_TEMPORAL_EXPORT const char* ck_temporal_last_error();
