
// Dans ce script, nous implementons de maniere locale, et a la compilation
// du programme, quelles boucles, parmis les 5 (en realite seulement 4 sont
// actuellement parallelisable) boucles exposees par la fonction gemm de BLIS,
// se verront parallelisees.

// Il est important de preciser que si, comme dans les scipts perf_BLIS_ENV_*.c,
// on choisit de confirgurer la parallelisation de maniere automatique, si une
// implementation manuelle de la strategie de parallelisation est egalement
// definie, c'est cette derniere qui s'impose.

// Avant d'executer ce script, comme pour le script d'implementation automatique
// globale pour variable d'environement, il faut definir, avant l'execution du
// programme compile, les variables d'environement suivantes
//
// 		$export GOMP_CPU_AFFINITY="0 1 2 3 4 5" #(Pour un ordinateur a 6 coeurs)
// OU
// 		$export OMP_PROC_BIND=close;export OMP_PLACES=cores

// Cette strategie de parallelisation expose chacunes des 5 boucles utilisee par
// la fonction bli_gemm_ex avec la fonction :
//
// void bli_thread_set_ways( dim_t jc, dim_t pc, dim_t ic, dim_t jr, dim_t ir );
//
// Plusieurs strategies etant parametrable. Nous pouvons en effet preciser
// les boucles qui seront parallelisees et le degres de parallelisation utilise.
// Nous serons donc amenes a essayer plusieurs configurations. Elles seront
// lors de l'execution du script.
// Notons que la parallelisation de fait de maniere herarchique, et que le
// nombre de thread employes correspond au produit ( jc * pc * ic * jr * ir ),
// chaque variable indique le nombre de thread sur lequels s'executent la boucle.
//

// A noter que la contribution principale apportee par la fonction gemm de BLIS
// est l'exposition de 2 boucles suppelementaire dans le micro-kernel historique
// de BLAS (correspondant a ir et jr).

#include <stdio.h>
#include <time.h>
#include "blis.h"


int main( int argc, char** argv )
{
	// Defition de la variable specifiant la parallelisation a effetuer
	rntm_t rntm;
	// Initialisation de la variables
	bli_rntm_init( &rntm );

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

	// Nous travaillerons avec des float (32 bits)
	dt = BLIS_FLOAT;

	// Variable de taille maximale d'une matrice et du nombre de boucle
	int N = 3050;
	int B = 5;


	printf( "## Script reposant sur l'activation LOCAL et SPECIFIQUE de la strategie\
 de parallelisation de BLIS ##\n\n");

	// PREMIERE STRATEGIE DE PARALLELISATION :
	//  jc = 2,  pc = 1,  ic = 1,  jr = 1,  ir = 3
	bli_rntm_set_ways( 2, 1, 1, 1, 3, &rntm );
	printf( "Temps d'execution moyen d'un matrice de taille :\n\n\
	\t-> Strategie de parallelisation : jc = 2, pc = 1, ic = 1, jr = 1, ir = 3\n\n" );

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
		//
		// Cette fois-ci l'utilisation de la variable rntm necessite d'utiliser
		// la version "experte" de bli_gemm, bli_gemm_ex, qui expose la
		// parametrisation de 2 arguments supplementaires, cntx_t (contexte qui
		// sera fixe a NULL) et rntm_t. C'est ce dernier qui nous interesse.
		elapsed = 0.0;
		for ( int j = 0; j < B; j++)
		{
			clock_gettime( CLOCK_MONOTONIC, &start );
			bli_gemm_ex( alpha, &a, &b, beta, &c, NULL, &rntm );
			clock_gettime( CLOCK_MONOTONIC, &finish );
			elapsed += ( finish.tv_sec - start.tv_sec );
			elapsed += ( finish.tv_nsec - start.tv_nsec ) / 1000000000.0;
		}
		printf( "\t- (%dx%d)\t:\t%f s\n", ( int )i, ( int )i, elapsed / B );

		// Liberation de l'espace memoire
		bli_obj_free( &a );
		bli_obj_free( &b );
		bli_obj_free( &c );

	}

	// DEUXIEME STRATEGIE DE PARALLELISATION :
	//  jc = 6,  pc = 1,  ic = 1,  jr = 1,  ir = 1
	bli_rntm_set_ways( 6, 1, 1, 1, 1, &rntm );
	printf( "Temps d'execution moyen d'un matrice de taille :\n\n\
	\t-> Strategie de parallelisation : jc = 6, pc = 1, ic = 1, jr = 1, ir = 1\n\n" );

	for ( dim_t i = 50; ( int )i < N; i += 100 )
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
		//
		// Cette fois-ci l'utilisation de la variable rntm necessite d'utiliser
		// la version "experte" de bli_gemm, bli_gemm_ex, qui expose la
		// parametrisation de 2 arguments supplementaires, cntx_t (contexte qui
		// sera fixe a NULL) et rntm_t. C'est ce dernier qui nous interesse.
		elapsed = 0.0;
		for ( int j = 0; j < B; j++)
		{
			clock_gettime( CLOCK_MONOTONIC, &start );
			bli_gemm_ex( alpha, &a, &b, beta, &c, NULL, &rntm );
			clock_gettime( CLOCK_MONOTONIC, &finish );
			elapsed += ( finish.tv_sec - start.tv_sec );
			elapsed += ( finish.tv_nsec - start.tv_nsec ) / 1000000000.0;
		}
		printf( "\t- (%dx%d)\t:\t%f s\n", ( int )i, ( int )i, elapsed / B );

		// Liberation de l'espace memoire
		bli_obj_free( &a );
		bli_obj_free( &b );
		bli_obj_free( &c );

	}

	// TROISIEME STRATEGIE DE PARALLELISATION :
	//  jc = 1,  pc = 1,  ic = 6,  jr = 1,  ir = 1
	bli_rntm_set_ways( 1, 1, 6, 1, 1, &rntm );
	printf( "Temps d'execution moyen d'un matrice de taille :\n\n\
	\t-> Strategie de parallelisation : jc = 1, pc = 1, ic = 6, jr = 1, ir = 1\n\n" );

	for ( dim_t i = 50; ( int )i < N; i += 100 )
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
		//
		// Cette fois-ci l'utilisation de la variable rntm necessite d'utiliser
		// la version "experte" de bli_gemm, bli_gemm_ex, qui expose la
		// parametrisation de 2 arguments supplementaires, cntx_t (contexte qui
		// sera fixe a NULL) et rntm_t. C'est ce dernier qui nous interesse.
		elapsed = 0.0;
		for ( int j = 0; j < B; j++)
		{
			clock_gettime( CLOCK_MONOTONIC, &start );
			bli_gemm_ex( alpha, &a, &b, beta, &c, NULL, &rntm );
			clock_gettime( CLOCK_MONOTONIC, &finish );
			elapsed += ( finish.tv_sec - start.tv_sec );
			elapsed += ( finish.tv_nsec - start.tv_nsec ) / 1000000000.0;
		}
		printf( "\t- (%dx%d)\t:\t%f s\n", ( int )i, ( int )i, elapsed / B );

		// Liberation de l'espace memoire
		bli_obj_free( &a );
		bli_obj_free( &b );
		bli_obj_free( &c );

	}

	// QUATRIEME STRATEGIE DE PARALLELISATION :
	//  jc = 1,  pc = 1,  ic = 1,  jr = 6,  ir = 1
	bli_rntm_set_ways( 1, 1, 1, 6, 1, &rntm );
	printf( "Temps d'execution moyen d'un matrice de taille :\n\n\
	\t-> Strategie de parallelisation : jc = 1, pc = 1, ic = 1, jr = 6, ir = 1\n\n" );

	for ( dim_t i = 50; ( int )i < N; i += 100 )
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
		//
		// Cette fois-ci l'utilisation de la variable rntm necessite d'utiliser
		// la version "experte" de bli_gemm, bli_gemm_ex, qui expose la
		// parametrisation de 2 arguments supplementaires, cntx_t (contexte qui
		// sera fixe a NULL) et rntm_t. C'est ce dernier qui nous interesse.
		elapsed = 0.0;
		for ( int j = 0; j < B; j++)
		{
			clock_gettime( CLOCK_MONOTONIC, &start );
			bli_gemm_ex( alpha, &a, &b, beta, &c, NULL, &rntm );
			clock_gettime( CLOCK_MONOTONIC, &finish );
			elapsed += ( finish.tv_sec - start.tv_sec );
			elapsed += ( finish.tv_nsec - start.tv_nsec ) / 1000000000.0;
		}
		printf( "\t- (%dx%d)\t:\t%f s\n", ( int )i, ( int )i, elapsed / B );

		// Liberation de l'espace memoire
		bli_obj_free( &a );
		bli_obj_free( &b );
		bli_obj_free( &c );

	}

	// CINQUIEME STRATEGIE DE PARALLELISATION :
	//  jc = 1,  pc = 1,  ic = 1,  jr = 1,  ir = 6
	bli_rntm_set_ways( 1, 1, 1, 1, 6, &rntm );
	printf( "Temps d'execution moyen d'un matrice de taille :\n\n\
	\t-> Strategie de parallelisation : jc = 1, pc = 1, ic = 1, jr = 1, ir = 6\n\n" );

	for ( dim_t i = 50; ( int )i < N; i += 100 )
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
		//
		// Cette fois-ci l'utilisation de la variable rntm necessite d'utiliser
		// la version "experte" de bli_gemm, bli_gemm_ex, qui expose la
		// parametrisation de 2 arguments supplementaires, cntx_t (contexte qui
		// sera fixe a NULL) et rntm_t. C'est ce dernier qui nous interesse.
		elapsed = 0.0;
		for ( int j = 0; j < B; j++)
		{
			clock_gettime( CLOCK_MONOTONIC, &start );
			bli_gemm_ex( alpha, &a, &b, beta, &c, NULL, &rntm );
			clock_gettime( CLOCK_MONOTONIC, &finish );
			elapsed += ( finish.tv_sec - start.tv_sec );
			elapsed += ( finish.tv_nsec - start.tv_nsec ) / 1000000000.0;
		}
		printf( "\t- (%dx%d)\t:\t%f s\n", ( int )i, ( int )i, elapsed / B );

		// Liberation de l'espace memoire
		bli_obj_free( &a );
		bli_obj_free( &b );
		bli_obj_free( &c );

	}

	// CINQUIEME STRATEGIE DE PARALLELISATION :
	//  jc = 1,  pc = 1,  ic = 1,  jr = 1,  ir = 6
	bli_rntm_set_ways( 1, 1, 3, 2, 1, &rntm );
	printf( "TEST :\n\n");

	for ( dim_t i = 50; ( int )i < N; i += 100 )
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
		//
		// Cette fois-ci l'utilisation de la variable rntm necessite d'utiliser
		// la version "experte" de bli_gemm, bli_gemm_ex, qui expose la
		// parametrisation de 2 arguments supplementaires, cntx_t (contexte qui
		// sera fixe a NULL) et rntm_t. C'est ce dernier qui nous interesse.
		elapsed = 0.0;
		for ( int j = 0; j < B; j++)
		{
			clock_gettime( CLOCK_MONOTONIC, &start );
			bli_gemm_ex( alpha, &a, &b, beta, &c, NULL, &rntm );
			clock_gettime( CLOCK_MONOTONIC, &finish );
			elapsed += ( finish.tv_sec - start.tv_sec );
			elapsed += ( finish.tv_nsec - start.tv_nsec ) / 1000000000.0;
		}
		printf( "\t- (%dx%d)\t:\t%f s\n", ( int )i, ( int )i, elapsed / B );

		// Liberation de l'espace memoire
		bli_obj_free( &a );
		bli_obj_free( &b );
		bli_obj_free( &c );

	}

}
