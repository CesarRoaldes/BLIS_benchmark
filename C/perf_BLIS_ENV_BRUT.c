#include <stdio.h>
#include <time.h>
#include "blis.h"

// Script sortant les resultats bruts destines a etre par la suite recuperes
// depuis Python.

// Cette mesure utilisera les deux paramtrisations possibles des affinites entre
// coeurs et threads.
// Pour plus d'information :
// https://github.com/flame/blis/blob/master/docs/Multithreading.md#specifying-thread-to-core-affinity


// L'enregistrement se fait a l'aide de la redirection de flux de la fonction
// printf depuis la CLI :

//  - Si on parametrise l'affinite entre les coeurs et les threads avec
// 	  GOMP_CPU_AFFINITY + OMP_PLACES :
//    	$touch pref_BLIS_ENV_OMP_OPT.txt
//	- Si on utilise une parametrisation plus fine a l'aide de l'option
//	  GOMP_CPU_AFFINITY :
// 	  	$touch pref_BLIS_ENV_AFF_OPT.txt

// 	puis (apres avoir compile a l'aide de make) :
//    	$./perf_BLIS_ENV_BRUT.x > pref_BLIS_ENV_??.txt
//  ou "??" correspond a OMP_OPT / AFF_OPT en fonction de la premiere etape.

int main( int argc, char** argv )
{
	// Attributs d'une matrice
	num_t dt;
	dim_t m, n, k;
	inc_t rs, cs;

	// Trois matrices (a, b, c) sont crees ainsi que deux sclaires (alpha, beta)
	obj_t a, b, c;
	obj_t* alpha;
	obj_t* beta;

	//  Variables permettant la mesure du temps
	struct timespec start, finish;
	double elapsed;

	// Variable de taille maximale d'une matrice et du nombre de boucle
	int N = 5050;
	int B = 5;


	// Nous travaillerons avec des float (32 bits)
	dt = BLIS_FLOAT;

	printf( "{" );
	for ( dim_t i = 50; ( int )i < N; i += 50 )
	{
		// Initialisation des 3 matrices carrees de taille (i x i)
		m = i; n = i; k = i; rs = 0; cs = 0;
		bli_obj_create( dt, m, n, rs, cs, &c );
		bli_obj_create( dt, m, k, rs, cs, &a );
		bli_obj_create( dt, k, n, rs, cs, &b );

		// On initialise les scalaires a 1
		alpha = &BLIS_ONE;
		beta  = &BLIS_ONE;

		// On met des valeurs aleatoires dans la matrice a et b
		bli_randm( &a );
		bli_randm( &b );
		// La matrice c est initialisee a zero
		bli_setm( &BLIS_ZERO, &c );

		// Calcul effectue par bli_gemm -> c := beta * c + alpha * a * b.
		// On boucle sur la fonction pour avoir son temps d'execution moyen
		// pour une taille de matrice i donnee.
		elapsed = 0.0;
		for ( int j = 0; j < B; j++)
		{
			clock_gettime( CLOCK_MONOTONIC, &start );
			bli_gemm( alpha, &a, &b, beta, &c );
			clock_gettime( CLOCK_MONOTONIC, &finish );
			elapsed += ( finish.tv_sec - start.tv_sec );
			elapsed += ( finish.tv_nsec - start.tv_nsec ) / 1000000000.0;
		}
		printf( "%d: %f,\n", ( int )i, elapsed / B );

		// Liberation de l'espace memoire
		bli_obj_free( &a );
		bli_obj_free( &b );
		bli_obj_free( &c );

	}

	printf( "}" );
}
