// Code source
// Progamme pour raspberry Pi 3B
// Détection via camera Pi (raspiCam)
// Module movidius et fichier personnel "graph" MobileNetSSD.
// Jean-Pierre Olemans 2019 janvier.
// ----------------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include <mvnc.h>
#include <fstream>
#include <iostream>
#include <raspicam/raspicam_cv.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <unistd.h>
#include <pthread.h>
#include <string.h>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <string>

struct ncDeviceHandle_t *deviceHandle;
struct ncGraphHandle_t *graphHandle;
        struct ncTensorDescriptor_t inputTensorDesc;
        struct ncTensorDescriptor_t outputTensorDesc;
struct ncFifoHandle_t * bufferIn;
struct ncFifoHandle_t * bufferOut;
unsigned int graphFileLen;
#define GRAPH_FILE_NAME "graph" //compile par mvNCCompile deploy.prototxt -w Mobilenet_iter_73000.caffemodel -s 12 -o graph

raspicam::RaspiCam_Cv Camera;
using namespace std;
using namespace cv;
string Id[3]={"rien","chat","chien"};
	      
//----------------------------------------------------------------------
void *LoadFile(const char *path, unsigned int *length)
{
	FILE *fp;
	char *buf;

	fp = fopen(path, "rb");
	if(fp == NULL)
		return 0;
	fseek(fp, 0, SEEK_END);
	*length = ftell(fp);
	rewind(fp);
	if(!(buf = (char*) malloc(*length)))
	{
		fclose(fp);
		return 0;
	}
	if(fread(buf, 1, *length, fp) != *length)
	{
		fclose(fp);
		free(buf);
		return 0;
	}
	fclose(fp);
	return buf;
}
//**********************************************************************
bool mvnc_create()
{
    ncStatus_t retCode;
    retCode = ncDeviceCreate(0,&deviceHandle);
    if(retCode != NC_OK)
       {
        printf("Error Create Handle movidius\n");
        return false;
        }
    printf("Create Handle movidius ok\n");    
    return true;
}
bool mvnc_open()
{
    ncStatus_t retCode;
    retCode = ncDeviceOpen(deviceHandle);
    if(retCode != NC_OK)
       {
        printf("Error open movidius\n");
        return false;
        }   
    printf("Open movidius device ok\n");
    return true;
}
bool mvnc_close()
{
    ncStatus_t retCode;    
    retCode = ncDeviceClose(deviceHandle);
    deviceHandle = NULL;
    if (retCode != NC_OK)
       {
        printf("Error close movidius device\n");           
        return false;
       }
    printf("close movidius device ok\n");       
    return true;
}
//----------------------------------------------------------------------
bool Create_Graph(void *graphFileBuf)
{
    ncStatus_t retCode = ncGraphCreate("graph", &graphHandle);
    if (retCode != NC_OK)
    {
        printf("Error - Create Graph Handle\n");
        return false;
    }
    printf("Create Graph handle ok\n");
    retCode = ncGraphAllocate(deviceHandle, graphHandle, graphFileBuf, graphFileLen);
    if (retCode != NC_OK)
      {
        printf("Error - Allocate Graph buffer\n");
        return false;
      }
    printf("Allocate Graph Buffer ok\n");
    return true;
}
//----------------------------------------------------------------------
bool Create_fifo(unsigned int *length)
{
ncStatus_t retCode;  
	*length = sizeof(struct ncTensorDescriptor_t);
        ncGraphGetOption(graphHandle, NC_RO_GRAPH_INPUT_TENSOR_DESCRIPTORS, &inputTensorDesc,  &*length);
        ncGraphGetOption(graphHandle, NC_RO_GRAPH_OUTPUT_TENSOR_DESCRIPTORS, &outputTensorDesc,  &*length);
        int dataTypeSize = outputTensorDesc.totalSize/(outputTensorDesc.w* outputTensorDesc.h*outputTensorDesc.c*outputTensorDesc.n);
         retCode = ncFifoCreate("FifoIn0", NC_FIFO_HOST_WO, &bufferIn);
        if (retCode != NC_OK)
            {
            printf("Error - Input Fifo Initialization failed!");
            return false;
            }
        retCode = ncFifoAllocate(bufferIn, deviceHandle, &inputTensorDesc, 2);
        if (retCode != NC_OK)
           {
            printf("Error - Input Fifo allocation failed!");
            return false;
           }
        retCode = ncFifoCreate("FifoOut0", NC_FIFO_HOST_RO, &bufferOut);
        if (retCode != NC_OK)
           {
            printf("Error - Output Fifo Initialization failed!");
            return false;
           }
        retCode = ncFifoAllocate(bufferOut, deviceHandle, &outputTensorDesc, 2);
        if (retCode != NC_OK)
           {
            printf("Error - Output Fifo allocation failed!");
            return false;
           }  
	printf("fifo ok");
return true;	   
}
//----------------------------------------------------------------------
bool Write_Element(Mat im ,unsigned int imageSize)
{
 //lecture image et insertion dans le module 
ncStatus_t retCode;
int i;
float *image;
Mat imresize(300,300,CV_8UC3);
	resize(im, imresize, imresize.size(), 0, 0, INTER_LINEAR);
	imageSize=sizeof(*image)*300*300*3;
	image = (float *) malloc(imageSize);
	if(!image) printf("Erreur allocation buffer image\n");
	for(i=0;i<300*300*3;i=i+3) 
	   {
	     image[i+0]=float((imresize.data[i+2]-127.5)*0.007843);
	     image[i+1]=float((imresize.data[i+1]-127.5)*0.007843);
	     image[i+2]=float((imresize.data[i+0]-127.5)*0.007843);	     
	   }
	//printf("Buffer image ok\n");
        retCode = ncFifoWriteElem(bufferIn, image, &imageSize, 0);
        free(image);
        if (retCode != NC_OK)
           {
            printf("Error - Failed to write element to Fifo!");
            return false;
           }
        // queue inference
        retCode = ncGraphQueueInference(graphHandle, &bufferIn, 1, &bufferOut, 1);
            if (retCode != NC_OK)
               {
                printf("Error - Failed to queue Inference!");
                return false;
               }
return true;	   
}
//----------------------------------------------------------------------
int main(int argc, char** argv)
{
Mat im;
int width,height; //Size picture display
bool fin;
unsigned int imageSize;
int X1,X2,Y1,Y2;
    namedWindow("MyWindow",WINDOW_AUTOSIZE);                //Window principale
    Camera.set(CV_CAP_PROP_FORMAT, CV_8UC3);
    Camera.set(CV_CAP_PROP_FRAME_WIDTH,640);
    Camera.set(CV_CAP_PROP_FRAME_HEIGHT,480);
    int loglevel = 2;
    ncStatus_t retCode = ncGlobalSetOption(NC_RW_LOG_LEVEL, &loglevel, sizeof(loglevel));
    if(!mvnc_create()) return 0;
    if(!mvnc_open  ()) return 0;
    
    void* graphFileBuf = LoadFile(GRAPH_FILE_NAME, &graphFileLen);//lecture graphHandle et allocation
    if(!Create_Graph(graphFileBuf)) 
        {
	mvnc_close();
	return 0;
	}
    unsigned int length;
    if(!Create_fifo(&length)) return 0;
    Camera.open();
    fin=false;
    Nodetect=0;
    while(!fin)
    {    
        Camera.grab();
        Camera.retrieve(im);
	width=im.cols;
	height=im.rows;
	if(!Write_Element(im,imageSize)) return 0;//insertion image dans le module
        unsigned int outputDataLength;
        length = sizeof(unsigned int);
        retCode = ncFifoGetOption(bufferOut, NC_RO_FIFO_ELEMENT_DATA_SIZE, &outputDataLength, &length);
        if (retCode || length != sizeof(unsigned int))
	       {
                printf("ncFifoGetOption failed, rc=%d\n", retCode);
                exit(-1);
               }
        void *result = malloc(outputDataLength);
        if (!result) 
	       {
                printf("malloc failed!\n");
                exit(-1);
               }
        void *userParam;
        retCode = ncFifoReadElem(bufferOut, result, &outputDataLength, &userParam);
        if (retCode != NC_OK)
               {
                printf("Error - Read Inference result failed!");
                exit(-1);
               }
        unsigned int numResults =  outputDataLength / sizeof(float);
	float *fresult = (float*) result;       
        for (int i = 0; i < 71 ; i=i+7)
                {
//	       printf("0=%f 1=%f 2=%f 3=%f 4=%f 5=%f 6=%f \n",fresult[i],fresult[i+1],fresult[i+2],fresult[i+3],fresult[i+4],fresult[i+5],fresult[i+6]);
	       if((fresult[i]==0.0)&&(fresult[i+2]>0.95)&&(fresult[i+2]<=1.0))//resultat valide et probabilite > 90%  
		  {
		    string name;
		    int index=int(fresult[i+1]);
		    name=Id[index];
		    X1=int(fresult[i+3]*width);
		    Y1=int(fresult[i+4]*height);
		    X2=int(fresult[i+5]*width);
		    Y2=int(fresult[i+6]*height);
	            printf("0=%f Index=%d Prob=%f X1=%d Y1=%d X2=%d Y2=%d  %s \n",fresult[i],index,fresult[i+2]*100,X1,Y1,X2,Y2,name.c_str());
	            rectangle(im, Rect (X1,Y1,X2,Y2), Scalar::all(180), 1, 8,0 );
		    putText(im,name, Point(X1,Y1), FONT_HERSHEY_SCRIPT_SIMPLEX, 1,Scalar::all(180), 3, 2);
		  }
                }
	free(result);
        imshow("MyWindow",im);		

        while (waitKey(1) == (char) 27) fin=true;
        }
   retCode = ncGraphDestroy(&graphHandle);
   if (retCode != NC_OK)
           {
            printf("Error - Failed to deallocate graph!");
            exit(-1);
           }
   retCode = ncFifoDestroy(&bufferOut);
   if (retCode != NC_OK)
           {
            printf("Error - Failed to deallocate fifo!");
            exit(-1);
           }

   retCode = ncFifoDestroy(&bufferIn);
   if (retCode != NC_OK)
        {
            printf("Error - Failed to deallocate fifo!");
            exit(-1);
        }
    graphHandle = NULL;
    free(graphFileBuf);
    if(!mvnc_close ()) return 0;    
    return 0;
}
