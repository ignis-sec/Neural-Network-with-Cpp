#include <stdio.h>
#include <stdlib.h>



int main()
{
	FILE *fptr;
	fptr = fopen("Data.txt","wb+");
	double a,b,c;
	int i=3000;
	fprintf(fptr,"topology: 3 2 4 1\n");
	while(i>0)
	{
		a=(double)rand()/(double)RAND_MAX/3.0;
		b=(double)rand()/(double)RAND_MAX/3.0;
		c=(double)rand()/(double)RAND_MAX/3.0;
		fprintf(fptr,"in: %.4lf %.4lf %.4lf \nout: %.4f\n",a,b,c,a+b+c);

		//printf("in: %d.0 %d.0 ",a,b);
		//if(a>b) fprintf(fptr,"out: 1.0"); else fprintf(fptr,"out: 0.0");
		i--;
	}
	fclose(fptr);
	return 0;
}
