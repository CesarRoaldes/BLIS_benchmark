
.PHONY: all bin clean run



#
# --- Determine makefile fragment location -------------------------------------
#

# Comments:
# - DIST_PATH is assumed to not exist if BLIS_INSTALL_PATH is given.
# - We must use recursively expanded assignment for LIB_PATH and INC_PATH in
#   the second case because CONFIG_NAME is not yet set.
LIB_PATH   := /usr/local/lib
INC_PATH   := /usr/local/include/blis
SHARE_PATH := /usr/local/share/blis

#ifneq ($(strip $(BLIS_LIB_PATH)),)
#LIB_PATH   := $(BLIS_LIB_PATH)
#endif
#
#ifneq ($(strip $(BLIS_INC_PATH)),)
#INC_PATH   := $(BLIS_INC_PATH)
#endif
#
#ifneq ($(strip $(BLIS_SHARE_PATH)),)
#SHARE_PATH := $(BLIS_SHARE_PATH)
#endif



#
# --- Include common makefile definitions --------------------------------------
#

# Include the common makefile fragment.
-include $(SHARE_PATH)/common.mk



#
# --- General build definitions ------------------------------------------------
#

TEST_SRC_PATH  := .
TEST_OBJ_PATH  := .

# Gather all local object files.
TEST_OBJS      := $(sort $(patsubst $(TEST_SRC_PATH)/%.c, \
                                    $(TEST_OBJ_PATH)/%.o, \
                                    $(wildcard $(TEST_SRC_PATH)/*.c)))

# Override the value of CINCFLAGS so that the value of CFLAGS returned by
# get-user-cflags-for() is not cluttered up with include paths needed only
# while building BLIS.
CINCFLAGS      := -I$(INC_PATH)

# Use the "framework" CFLAGS for the configuration family.
CFLAGS         := $(call get-user-cflags-for,$(CONFIG_NAME))

# Add local header paths to CFLAGS
CFLAGS         += -I$(TEST_SRC_PATH)

# Locate the libblis library to which we will link.
#LIBBLIS_LINK   := $(LIB_PATH)/$(LIBBLIS_L)

# Binary executable name.
TEST_BINS      := perf_BLIS_ENV.x \
				  perf_BLIS_ENV_BRUT.x \
				  perf_BLIS_LOCAL_RUNTIME.x \

#
# --- Targets/rules ------------------------------------------------------------
#

# --- Primary targets ---

all: bin

bin: $(TEST_BINS)


# --- Environment check rules ---

check-env: check-env-make-defs check-env-fragments check-env-config-mk

check-env-config-mk:
ifeq ($(CONFIG_MK_PRESENT),no)
    $(error Cannot proceed: config.mk not detected! Run configure first)
endif

check-env-make-defs: check-env-fragments
ifeq ($(MAKE_DEFS_MK_PRESENT),no)
    $(error Cannot proceed: make_defs.mk not detected! Invalid configuration)
endif


# --Object file rules --

$(TEST_OBJ_PATH)/%.o: $(TEST_SRC_PATH)/%.c $(LIBBLIS_LINK)
ifeq ($(ENABLE_VERBOSE),yes)
	$(CC) $(CFLAGS) -c $< -o $@
else
	@echo "Compiling $@"
	@$(CC) $(CFLAGS) -c $< -o $@
endif


# -- Executable file rules --

%.x: %.o $(LIBBLIS_LINK)
ifeq ($(ENABLE_VERBOSE),yes)
	$(LINKER) $< $(LIBBLIS_LINK) $(LDFLAGS) -o $@
else
	@echo "Linking $@ against '$(LIBBLIS_LINK) $(LDFLAGS)'"
	@$(LINKER) $< $(LIBBLIS_LINK) $(LDFLAGS) -o $@
endif

# -- Test run rules --

#run: $(TEST_BIN)
#	./$(TEST_BIN)

# -- Clean rules --

clean:
	- $(RM_F) $(TEST_OBJS) $(TEST_BINS)
