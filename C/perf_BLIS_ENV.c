#include <stdio.h>
#include <time.h>
#include "blis.h"

// Ce script presente une configuration gloabal du multithreading de l'API BLIS
// par l'utilisation de variables d'environement definie avant l'execution du
// de l'executable découlant de fichier source, perf_BLIS_ENV.x .

// Plusieurs options sont disponibles pour activer le multithreading. Cependant,
// une etape prealable a chacunes d'entre elles est l'activation de variables
// d'environement avant l'execution de l'executable .x :
//
// 		$export GOMP_CPU_AFFINITY="0 1 2 3 4 5" #(Pour un ordinateur a 6 coeurs)
//
// OU
//
// 		$export OMP_PROC_BIND=close;export OMP_PLACES=core


// L'autorisation du multithreading peut parametree en utilisant les variables
// globales AVANT le lancement du script, c'est cette methode qu'utilise ce
// script :

// Ici, la strategie de multithreading deployee par BLIS enclanchee par la
// configuration de la variable d'environement BLIS_CPU_THREADS. La commande
// associee a cette activation est donnee ci dessous (sur une machine disposant)
// de 6 coeurs) :
//
// 		$export BLIS_CPU_THREADS=6
//
// Cette methode est dite automatique car la strategie est mise en place a
// l'aide d'heuristiques determinant une strategie de parallelisation dite
// "resonables".

// Si ces variables d'environement ne sont pas declarees, le programme s'execute
// sans paralleliser les calcule impliques dans l'etape de multiplication de
// matrice de la fonction gemm. Les resultats obtenus dans sans parallelisation
// sont rapportes dans le fichier perf_BLIS_SINGLE_THREAD.txt .


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

	printf( "## Resultats d'une activation globale et automatique de BLIS\n\
par l'utilisation de la variable d'environement BLIS_CPU_THREADS ##\n\
(Voir commentaires du fichier sources)\n\n");
	printf( "Temps d'execution moyen d'un matrice de taille :\n\n" );

	// Nous travaillerons avec des float (32 bits)
	dt = BLIS_FLOAT;


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
		printf( "\t- (%dx%d)\t:\t%f s\n", ( int )i, ( int )i, elapsed / B );

		// Liberation de l'espace memoire
		bli_obj_free( &a );
		bli_obj_free( &b );
		bli_obj_free( &c );

	}

}
