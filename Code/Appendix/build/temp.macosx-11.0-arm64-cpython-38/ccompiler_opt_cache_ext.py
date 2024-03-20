# AUTOGENERATED DON'T EDIT
# Please make changes to the code generator             (distutils/ccompiler_opt.py)
hash = 2491545691
data = \
{'cache_infile': False,
 'cache_me': {"('cc_test_flags', ['-O3'])": False,
              "('cc_test_flags', ['-Werror'])": False,
              "('cc_test_flags', ['-march=native'])": False,
              "('feature_flags', 'NEON')": [],
              "('feature_flags', 'NEON_VFPV4')": [],
              "('feature_flags', set())": [],
              "('feature_is_supported', 'ASIMD', 'force_flags', 'macros', None, [])": False,
              "('feature_is_supported', 'ASIMDDP', 'force_flags', 'macros', None, [])": False,
              "('feature_is_supported', 'ASIMDFHM', 'force_flags', 'macros', None, [])": False,
              "('feature_is_supported', 'ASIMDHP', 'force_flags', 'macros', None, [])": False,
              "('feature_is_supported', 'NEON', 'force_flags', 'macros', None, [])": False,
              "('feature_is_supported', 'NEON_FP16', 'force_flags', 'macros', None, [])": False,
              "('feature_is_supported', 'NEON_VFPV4', 'force_flags', 'macros', None, [])": False,
              "('feature_test', 'NEON', None, 'macros', [])": False,
              "('feature_test', 'NEON_VFPV4', None, 'macros', [])": False},
 'cache_private': {'sources_status'},
 'cc_flags': {'native': [], 'opt': [], 'werror': []},
 'cc_has_debug': True,
 'cc_has_native': False,
 'cc_is_cached': True,
 'cc_is_clang': True,
 'cc_is_gcc': False,
 'cc_is_icc': False,
 'cc_is_iccw': False,
 'cc_is_msvc': False,
 'cc_is_nocc': False,
 'cc_march': 'aarch64',
 'cc_name': 'clang',
 'cc_noopt': False,
 'cc_on_aarch64': True,
 'cc_on_armhf': False,
 'cc_on_noarch': False,
 'cc_on_ppc64': False,
 'cc_on_ppc64le': False,
 'cc_on_x64': False,
 'cc_on_x86': False,
 'feature_is_cached': True,
 'feature_min': {'NEON_VFPV4', 'NEON_FP16', 'NEON', 'ASIMD'},
 'feature_supported': {'ASIMD': {'autovec': True,
                                 'implies': ['NEON', 'NEON_FP16', 'NEON_VFPV4'],
                                 'implies_detect': False,
                                 'interest': 4},
                       'ASIMDDP': {'flags': ['-march=armv8.2-a+dotprod'],
                                   'implies': ['ASIMD'],
                                   'interest': 6},
                       'ASIMDFHM': {'flags': ['-march=armv8.2-a+fp16fml'],
                                    'implies': ['ASIMDHP'],
                                    'interest': 7},
                       'ASIMDHP': {'flags': ['-march=armv8.2-a+fp16'],
                                   'implies': ['ASIMD'],
                                   'interest': 5},
                       'NEON': {'autovec': True,
                                'headers': ['arm_neon.h'],
                                'implies': ['NEON_FP16', 'NEON_VFPV4', 'ASIMD'],
                                'interest': 1},
                       'NEON_FP16': {'autovec': True,
                                     'implies': ['NEON', 'NEON_VFPV4', 'ASIMD'],
                                     'interest': 2},
                       'NEON_VFPV4': {'autovec': True,
                                      'implies': ['NEON', 'NEON_FP16', 'ASIMD'],
                                      'interest': 3}},
 'hit_cache': False,
 'parse_baseline_flags': [],
 'parse_baseline_names': [],
 'parse_dispatch_names': [],
 'parse_is_cached': True,
 'parse_target_groups': {'SIMD_TEST': (True, [], [])},
 'sources_status': {}}