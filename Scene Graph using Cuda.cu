
/*
	CS 6023 Assignment 3.
	Do not make any changes to the boiler plate code or the other files in the folder.
	Use cudaFree to deallocate any memory not in usage.
	Optimize as much as possible.
*/

#include "SceneNode.h"
#include <queue>
#include "Renderer.h"
#include <stdio.h>
#include <string.h>
#include <cuda.h>
#include <chrono>


__global__ void dkernel1(int limit,int *gstart,int *gt,int *gcount,int *gchild,int *gglobalx,int *gglobaly)
{
    unsigned id =blockIdx.x*blockDim.x+threadIdx.x;
    if(id<limit){
    int ver=gt[id*3];
    int di=gt[id*3+1];
    int n=gt[id*3+2];
    int childcount=gcount[ver];
    int s=gstart[ver];
    if(di==2)
    { 
     atomicSub(&gglobaly[ver],n);
    }
    else if(di==3)
    { 
      atomicAdd(&gglobaly[ver],n);
    }
    else if(di==1) //DOWN
    {
      atomicAdd(&gglobalx[ver],n);
    }
    else if(di==0) 
    {
       atomicSub(&gglobalx[ver],n);
    }
    for(int i=s;i<s+childcount;i++)
    {
        int f=gchild[i];
        if(di==2)
        { 
           atomicSub(&gglobaly[f],n);
        }
        else if(di==3)
        {
          atomicAdd(&gglobaly[f],n);
        }
        else if(di==1) 
        {
          atomicAdd(&gglobalx[f],n);
        }
        else if(di==0) 
        {
          atomicSub(&gglobalx[f],n);
        }
    }
    }
}

__global__ void dkernel2(int **gm,int *meshx,int *meshy,int *gx,int *gy,int *gopacity,int frameSizeX,int frameSizeY,int V,int *final)
{
    unsigned id=blockIdx.x*blockDim.x+threadIdx.x;
    if(id<(frameSizeX*frameSizeY))
    {
        int row=id/frameSizeY;
        int col=id%frameSizeY;
        int meshopacity=-1;
        bool x=false;
        for(int k=0;k<V;k++)
        {
            if(row<meshx[k]+gx[k] && row>=gx[k])
            {
                if(col<meshy[k]+gy[k] && col>=gy[k])
                {
                  x=meshopacity<gopacity[k];
                  if(x)
                  {
                      meshopacity=gopacity[k];
                      int mx=row-gx[k];
                      int my=col-gy[k];
                      final[row*frameSizeY+col]=gm[k][mx*meshy[k]+my];

                  }
                }
            }
        }
    }

}




void readFile (const char *fileName, std::vector<SceneNode*> &scenes, std::vector<std::vector<int> > &edges, std::vector<std::vector<int> > &translations, int &frameSizeX, int &frameSizeY) {
	/* Function for parsing input file*/

	FILE *inputFile = NULL;
	// Read the file for input.
	if ((inputFile = fopen (fileName, "r")) == NULL) {
		printf ("Failed at opening the file %s\n", fileName) ;
		return ;
	}

	// Input the header information.
	int numMeshes ;
	fscanf (inputFile, "%d", &numMeshes) ;
	fscanf (inputFile, "%d %d", &frameSizeX, &frameSizeY) ;


	// Input all meshes and store them inside a vector.
	int meshX, meshY ;
	int globalPositionX, globalPositionY; // top left corner of the matrix.
	int opacity ;
	int* currMesh ;
	for (int i=0; i<numMeshes; i++) {
		fscanf (inputFile, "%d %d", &meshX, &meshY) ;
		fscanf (inputFile, "%d %d", &globalPositionX, &globalPositionY) ;
		fscanf (inputFile, "%d", &opacity) ;
		currMesh = (int*) malloc (sizeof (int) * meshX * meshY) ;
		for (int j=0; j<meshX; j++) {
			for (int k=0; k<meshY; k++) {
				fscanf (inputFile, "%d", &currMesh[j*meshY+k]) ;
			}
		}
		//Create a Scene out of the mesh.
		SceneNode* scene = new SceneNode (i, currMesh, meshX, meshY, globalPositionX, globalPositionY, opacity) ;
		scenes.push_back (scene) ;
	}

	// Input all relations and store them in edges.
	int relations;
	fscanf (inputFile, "%d", &relations) ;
	int u, v ;
	for (int i=0; i<relations; i++) {
		fscanf (inputFile, "%d %d", &u, &v) ;
		edges.push_back ({u,v}) ;
	}

	// Input all translations.
	int numTranslations ;
	fscanf (inputFile, "%d", &numTranslations) ;
	std::vector<int> command (3, 0) ;
	for (int i=0; i<numTranslations; i++) {
		fscanf (inputFile, "%d %d %d", &command[0], &command[1], &command[2]) ;
		translations.push_back (command) ;
	}
}


void writeFile (const char* outputFileName, int *hFinalPng, int frameSizeX, int frameSizeY) {
	/* Function for writing the final png into a file.*/
	FILE *outputFile = NULL;
	if ((outputFile = fopen (outputFileName, "w")) == NULL) {
		printf ("Failed while opening output file\n") ;
	}

	for (int i=0; i<frameSizeX; i++) {
		for (int j=0; j<frameSizeY; j++) {
			fprintf (outputFile, "%d ", hFinalPng[i*frameSizeY+j]) ;
		}
		fprintf (outputFile, "\n") ;
	}
}


int main (int argc, char **argv) {

	// Read the scenes into memory from File.
	const char *inputFileName = argv[1] ;
	int* hFinalPng ;

	int frameSizeX, frameSizeY ;
	std::vector<SceneNode*> scenes ;
	std::vector<std::vector<int> > edges ;
	std::vector<std::vector<int> > translations ;
	readFile (inputFileName, scenes, edges, translations, frameSizeX, frameSizeY) ;
	hFinalPng = (int*) malloc (sizeof (int) * frameSizeX * frameSizeY) ;

	// Make the scene graph from the matrices.
    Renderer* scene = new Renderer(scenes, edges) ;

	// Basic information.
	int V = scenes.size () ;
	int E = edges.size () ;
	int numTranslations = translations.size () ;

	// Convert the scene graph into a csr.
	scene->make_csr () ; // Returns the Compressed Sparse Row representation for the graph.
	int *hOffset = scene->get_h_offset () ;
	int *hCsr = scene->get_h_csr () ;
	int *hOpacity = scene->get_opacity () ; // hOpacity[vertexNumber] contains opacity of vertex vertexNumber.
	int **hMesh = scene->get_mesh_csr () ; // hMesh[vertexNumber] contains the mesh attached to vertex vertexNumber.
	int *hGlobalCoordinatesX = scene->getGlobalCoordinatesX () ; // hGlobalCoordinatesX[vertexNumber] contains the X coordinate of the vertex vertexNumber.
	int *hGlobalCoordinatesY = scene->getGlobalCoordinatesY () ; // hGlobalCoordinatesY[vertexNumber] contains the Y coordinate of the vertex vertexNumber.
	int *hFrameSizeX = scene->getFrameSizeX () ; // hFrameSizeX[vertexNumber] contains the vertical size of the mesh attached to vertex vertexNumber.
	int *hFrameSizeY = scene->getFrameSizeY () ; // hFrameSizeY[vertexNumber] contains the horizontal size of the mesh attached to vertex vertexNumber.

	auto start = std::chrono::high_resolution_clock::now () ;


	// Code begins here.
	// Do not change anything above this comment.
  std::vector<int> child;
  std::vector<int> count;
  std::vector<int> nstart;
  nstart.push_back(0);
  int x=0;

  for (int i=0;i<V;i++)
  {
    int ncount=0;
    std::queue<int>q;
    q.push(i);
    while(!q.empty())
    {
      int x=q.front();
      q.pop();
      int start=hOffset[x];
      int diff=hOffset[x+1]-start;
      for(int j=start;j<start+diff;j++)
      {
        q.push(hCsr[j]);
        child.push_back(hCsr[j]);
        ncount++;
      }
    }
    x+=ncount;
    count.push_back(ncount);
    nstart.push_back(x);
  }
  //dkernel-1
  int *cchild=(int*)malloc(sizeof(int)*child.size());
  for(int i=0;i<child.size();i++)
  cchild[i]=child[i];
  int blocks=ceil(numTranslations/1024.0);
  int *gchild;
  cudaMalloc(&gchild,child.size()*sizeof(int));
  cudaMemcpy(gchild,cchild,child.size()*sizeof(int),cudaMemcpyHostToDevice);
  int *gglobalx;
  int *gglobaly;
  cudaMalloc(&gglobalx,V*sizeof(int));
  cudaMalloc(&gglobaly,V*sizeof(int));
  cudaMemcpy(gglobalx,hGlobalCoordinatesX,V*sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(gglobaly,hGlobalCoordinatesY,V*sizeof(int),cudaMemcpyHostToDevice);
  int *gcount;
  int *ccount=(int*)malloc(sizeof(int)*count.size());
  for(int i=0;i<count.size();i++)
  ccount[i]=count[i];
  cudaMalloc(&gcount,count.size()*sizeof(int));
  cudaMemcpy(gcount,ccount,count.size()*sizeof(int),cudaMemcpyHostToDevice);
  int *ct=(int*)malloc(numTranslations*3*sizeof(int));
  int k=0;
  for(int i=0;i<translations.size();i++)
  {
      for(int j=0;j<3;j++)
      ct[k++]=translations[i][j];
  }
  int *gt;
  cudaMalloc(&gt,sizeof(int)*numTranslations*3);
  cudaMemcpy(gt,ct,sizeof(int)*numTranslations*3,cudaMemcpyHostToDevice);
  int *gstart;
  int *cstart=(int*)malloc(sizeof(int)*nstart.size());
  for(int i=0;i<nstart.size();i++)
  cstart[i]=nstart[i];
  cudaMalloc(&gstart,nstart.size()*sizeof(int));
  cudaMemcpy(gstart,cstart,sizeof(int)*nstart.size(),cudaMemcpyHostToDevice);
  dkernel1<<<blocks,1024>>>(numTranslations,gstart,gt,gcount,gchild,gglobalx,gglobaly);
  cudaMemcpy(hGlobalCoordinatesX,gglobalx,V*sizeof(int),cudaMemcpyDeviceToHost);
  cudaMemcpy(hGlobalCoordinatesY,gglobaly,V*sizeof(int),cudaMemcpyDeviceToHost);
  free(cchild);
  free(cstart);
  free(ct);
  free(ccount);
  cudaFree(gchild);
  cudaFree(gstart);
  cudaFree(gt);
  cudaFree(gcount);

  //dkernel-2
  int **gm;
  cudaMalloc(&gm,V*sizeof(int*));
  for(int i=0;i<V;i++){
     int *dummy;
     cudaMalloc(&dummy,hFrameSizeX[i]*hFrameSizeY[i]*sizeof(int));
     cudaMemcpy(dummy,hMesh[i],hFrameSizeX[i]*hFrameSizeY[i]*sizeof(int),cudaMemcpyHostToDevice);
     cudaMemcpy(&gm[i],&dummy,sizeof(int*),cudaMemcpyHostToDevice);}
  int nblocks=ceil(frameSizeX*frameSizeY/1024.0);
  int *meshx;
  int *meshy;
  cudaMalloc(&meshx,V*sizeof(int));
  cudaMalloc(&meshy,V*sizeof(int));
  cudaMemcpy(meshx,hFrameSizeX,V*sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(meshy,hFrameSizeY,V*sizeof(int),cudaMemcpyHostToDevice);
  int *gx;
  int *gy;
  cudaMalloc(&gx,V*sizeof(int));
  cudaMalloc(&gy,V*sizeof(int));
  cudaMemcpy(gx,hGlobalCoordinatesX,V*sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(gy,hGlobalCoordinatesY,V*sizeof(int),cudaMemcpyHostToDevice);
  int *gopacity;
  cudaMalloc(&gopacity,V*sizeof(int));
  cudaMemcpy(gopacity,hOpacity,V*sizeof(int),cudaMemcpyHostToDevice);
  int *final;
  cudaMalloc(&final,frameSizeX*frameSizeY*sizeof(int));
  dkernel2<<<nblocks,1024>>>(gm,meshx,meshy,gx,gy,gopacity,frameSizeX,frameSizeY,V,final);
  cudaMemcpy(hFinalPng,final,frameSizeX*frameSizeY*sizeof(int),cudaMemcpyDeviceToHost);
 // Do not change anything below this comment.
 // Code ends here.

	auto end  = std::chrono::high_resolution_clock::now () ;

	std::chrono::duration<double, std::micro> timeTaken = end-start;

	printf ("execution time : %f\n", timeTaken) ;
	// Write output matrix to file.
	const char *outputFileName = argv[2] ;
	writeFile (outputFileName, hFinalPng, frameSizeX, frameSizeY) ;

}
