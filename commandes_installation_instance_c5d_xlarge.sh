# Machines utilisée c5d.xlarge avec ubuntu 16.04 LTS
########
cd ~/
git clone https://github.com/flame/blis.git


cd /tmp
url -O https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh
bash Anaconda3-2019.10-Linux-x86_64.sh
source ~/.bashrc

ldconfig -p | grep libpthread
cd ~/blis
sudo apt install build-essential
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt-get update
sudo apt-get install gcc-6 g++-6
sudo update-alternatives --install /usr/bin/gcc gcc /usr//bin//gcc-6 20 --slave /usr/bin//g++ g++ /usr/bin/g++-6

./configure --enable-threading=openmp --enable-cblas  auto

(sudo apt-get install libomp-dev
echo |cpp -fopenmp -dM |grep -i open
sudo apt install libomp-dev)

make -j 4
make check

cd ~/blis/frame/compat/cblas/src
cp cblas.h ~/blis/include/skx/cblas.h
cd ~/blis/
sudo make install
cd examples/oapi
export BLIS_INSTALL_PATH=/usr/local;
make
./00obj_basic.x

cp -r ../oapi/ ~/test/oapi
cd ~/test/oapi

vim test.c
    #include <stdio.h>
    #include <time.h>
    #include "blis.h"

    int main( int argc, char** argv )
    {
    	num_t dt;
    	dim_t m, n, k;
    	inc_t rs, cs;

    	obj_t a, b, c;
    	obj_t* alpha;
    	obj_t* beta;

    	struct timespec start, finish;
    	double elapsed;

    	printf( "\n#\n#  -- Exemple 1 --\n#\n\n" );
    	printf( "Debut de l'operation.\n\n" );
    	// Create some matrix operands to work with.
    	dt = BLIS_DOUBLE;
    	m = 10000; n = 10000; k = 10000; rs = 0; cs = 0;
    	bli_obj_create( dt, m, n, rs, cs, &c );
    	bli_obj_create( dt, m, k, rs, cs, &a );
    	bli_obj_create( dt, k, n, rs, cs, &b );

    	// Set the scalars to use.
    	alpha = &BLIS_ONE;
    	beta  = &BLIS_ONE;

    	// Initialize the matrix operands.
    	bli_randm( &a );
    	bli_setm( &BLIS_ONE, &b );
    	bli_setm( &BLIS_ZERO, &c );

    	//bli_printm( "a: randomized", &a, "%4.1f", "" );
    	//bli_printm( "b: set to 1.0", &b, "%4.1f", "" );
    	//bli_printm( "c: initial value", &c, "%4.1f", "" );

    	// c := beta * c + alpha * a * b, where 'a', 'b', and 'c' are general.
    	clock_gettime(CLOCK_MONOTONIC, &start);
    	bli_gemm( alpha, &a, &b, beta, &c );
    	clock_gettime(CLOCK_MONOTONIC, &finish);

    	//bli_printm( "c: after gemm", &c, "%4.1f", "" );
    	elapsed = (finish.tv_sec - start.tv_sec);
    	elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    	printf("FIN.\n\nTemps d'execution : %f ssecondes\n", elapsed);
    	// Free the objects.
    	bli_obj_free( &a );
    	bli_obj_free( &b );
    	bli_obj_free( &c );



    }

vim Makefile
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
    TEST_BINS      := test.x

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

make
./test.x

export OMP_PROC_BIND=close



libraries = blis
library_dirs = /home/ubuntu/blis/lib/skx
include_dirs = /home/ubuntu/blis/frame/compat/cblas/src
runtime_library_dirs = /home/ubuntu/blis/lib/skx
