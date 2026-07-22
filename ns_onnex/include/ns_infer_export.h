// ns_infer_export.h
// Symbol export/import macros for the ns_infer DLL.
//
// When building the DLL, define NS_INFER_EXPORTS to dllexport symbols.
// When consuming the DLL, leave it undefined so the macro resolves to dllimport.

#ifndef NS_INFER_EXPORT_H
#define NS_INFER_EXPORT_H

#if defined(_WIN32) || defined(_WIN64)
    #ifdef NS_INFER_EXPORTS
        #define NS_INFER_API __declspec(dllexport)
    #else
        #define NS_INFER_API __declspec(dllimport)
    #endif
#else
    #if defined(__GNUC__) && (__GNUC__ >= 4)
        #define NS_INFER_API __attribute__((visibility("default")))
    #else
        #define NS_INFER_API
    #endif
#endif

#endif // NS_INFER_EXPORT_H
