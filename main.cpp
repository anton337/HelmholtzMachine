#include <iostream>
#include <vector>
#include <windows.h>
#include <fstream>

#define NUM_THREADS 24


#include <GL/glut.h>


struct Operator
{

	std::size_t m;
	std::size_t n;

	std::vector < std::vector < double > > W;

	Operator ( std::size_t p_m , std::size_t p_n )
		: m ( p_m )
		, n ( p_n )
	{

	}

	std::vector < double > apply ( std::vector < double > const & x , double (*f) ( double ))
	{

		std::vector < double > y ( m , 0 );

		if ( x . size () != n )
		{
			std::cout << "input vector size mismatch" << std::endl;
			exit(0);
		}

		for ( std::size_t j(0)
			; j < m
			; ++j
			)
		{
			for ( std::size_t i(0)
				; i < n
				; ++i
				)
			{
				y[j] += W[j][i] * x[i];
			}
			y[j] = (*f) ( y[j] );
		}

	}

};


struct Network
{

	std::vector < std::size_t > sizes;

	std::vector < std::size_t > bias_sizes;

	std::vector < Operator * > operators;

	void init	( 
		  std::vector < std::size_t > const & sizes 
		, std::vector < std::size_t > const & bias_sizes
				)
	{
		for ( std::size_t k(0)
			; k+1 < sizes . size ()
			; ++k
			)
		{
			operators . push_back ( new Operator ( sizes[k+1] , sizes[k] + bias_sizes[k] ) );
		}
	}


};



struct Image
{
	std::vector < std::size_t > data;
};


std::vector < std::size_t > train_labels;
std::vector < Image > train_images;


std::size_t img_index = 0;

float img_index_flt = 0;


void
drawBox(void)
{
	img_index_flt += 0.2;
	img_index = (int)(img_index_flt) % train_images.size();
	double val;
	glBegin(GL_QUADS);
	if ( img_index < train_images . size () )
	{
		for ( std::size_t x(0)
			, k(0)
			; x < 28
			; ++x
			)
		{
			for ( std::size_t y(0)
				; y < 28
				; ++y
				, ++k
				)
			{
				val = (float)train_images[img_index].data[k]/256.0;
				glColor3f(val,val,val);
				glVertex3f( -1 + 2* y   /28.0 , -(-1 + 2* x   /28.0) , 0 );
				glVertex3f( -1 + 2* y   /28.0 , -(-1 + 2*(x+1)/28.0) , 0 );
				glVertex3f( -1 + 2*(y+1)/28.0 , -(-1 + 2*(x+1)/28.0) , 0 );
				glVertex3f( -1 + 2*(y+1)/28.0 , -(-1 + 2* x   /28.0) , 0 );
			}
		}
	}
	glEnd();
}

void idle()
{
	glutPostRedisplay();
}

void
display(void)
{
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  drawBox();
  glutSwapBuffers();
}

void
init(void)
{

  /* Use depth buffering for hidden surface elimination. */
  glEnable(GL_DEPTH_TEST);

  /* Setup the view of the cube. */
  glMatrixMode(GL_PROJECTION);
  gluPerspective( /* field of view in degree */ 40.0,
    /* aspect ratio */ 1.0,
    /* Z near */ 1.0, /* Z far */ 10.0);
  glMatrixMode(GL_MODELVIEW);
  gluLookAt(0.0, 0.0, 5.0,  /* eye is at (0,0,5) */
    0.0, 0.0, 0.0,      /* center is at (0,0,0) */
    0.0, 1.0, 0.);      /* up is in positive Y direction */

}

DWORD WINAPI ThreadFunc(void* data) {
	// Do stuff.  This will be the first function called on the new thread.
	// When this function returns, the thread goes away.  See MSDN for more details.
	std::cout << "Hello from thread: " << GetCurrentThreadId() << std::endl;
	while(1)
	{
		
	}
	return 0;
}

void read_binary_labels_file ( std::string filename , std::vector < std::size_t > & data )
{
	std::streampos size;
	char * memblock;
	std::ifstream file (filename.c_str(), std::ios::in|std::ios::binary|std::ios::ate);
	if (file.is_open())
	{
		size = file.tellg();
		memblock = new char [size];
		file.seekg (0, std::ios::beg);
		file.read (memblock, size);
		file.close();
		std::cout << "the entire file content is in memory" << std::endl;
		std::size_t magic_number =	  (int)(unsigned char)memblock[3]
									+ (int)(unsigned char)memblock[2]*256
									+ (int)(unsigned char)memblock[1]*256*256
									+ (int)(unsigned char)memblock[0]*256*256*256
									;
		std::size_t num_points =	  (int)(unsigned char)memblock[7]
									+ (int)(unsigned char)memblock[6]*256
									+ (int)(unsigned char)memblock[5]*256*256
									+ (int)(unsigned char)memblock[4]*256*256*256
									;
		std::cout << "magic number:" << magic_number << std::endl;
		std::cout << "num points:"   << num_points   << std::endl;
		for ( std::size_t k(0)
			; k < num_points
			; ++k
			)
		{
			data . push_back ( (std::size_t)(unsigned char)memblock[8+k] );
		}
		delete[] memblock;
	}
	else std::cout << "Unable to open file" << std::endl;
}


void read_binary_images_file ( std::string filename , std::vector < Image > & data )
{
	std::streampos size;
	char * memblock;
	std::ifstream file (filename.c_str(), std::ios::in|std::ios::binary|std::ios::ate);
	if (file.is_open())
	{
		size = file.tellg();
		memblock = new char [size];
		file.seekg (0, std::ios::beg);
		file.read (memblock, size);
		file.close();
		std::cout << "the entire file content is in memory" << std::endl;
		std::size_t magic_number =	  (int)(unsigned char)memblock[3]
									+ (int)(unsigned char)memblock[2]*256
									+ (int)(unsigned char)memblock[1]*256*256
									+ (int)(unsigned char)memblock[0]*256*256*256
									;
		std::size_t num_points =	  (int)(unsigned char)memblock[7]
									+ (int)(unsigned char)memblock[6]*256
									+ (int)(unsigned char)memblock[5]*256*256
									+ (int)(unsigned char)memblock[4]*256*256*256
									;
		std::size_t wx =			  (int)(unsigned char)memblock[11]
									+ (int)(unsigned char)memblock[10]*256
									+ (int)(unsigned char)memblock[9 ]*256*256
									+ (int)(unsigned char)memblock[8 ]*256*256*256
									;
		std::size_t wy =			  (int)(unsigned char)memblock[15]
									+ (int)(unsigned char)memblock[14]*256
									+ (int)(unsigned char)memblock[13]*256*256
									+ (int)(unsigned char)memblock[12]*256*256*256
									;

		std::cout << "magic number:" << magic_number << std::endl;
		std::cout << "num points:"   << num_points   << std::endl;
		std::cout << "wx:" << wx << std::endl;
		std::cout << "wy:" << wy << std::endl;
		for ( std::size_t k(0)
			, i(0)
			; k < num_points
			; ++k
			)
		{
			Image image;
			for ( std::size_t x(0)
				; x < wx
				; ++x
				)
			{
				for ( std::size_t y(0)
					; y < wy
					; ++y
					, ++i
					)
				{
					image . data . push_back ( (std::size_t)(unsigned char)memblock[16+i] );
				}
			}
			data . push_back ( image );
		}
		delete[] memblock;
	}
	else std::cout << "Unable to open file" << std::endl;
}

int main(int argc,char ** argv)
{
	std::cout << "Hello World!" << std::endl;

	read_binary_labels_file ( "C:\\Users\\H181523\\Downloads\\train-labels.idx1-ubyte" , train_labels );
	read_binary_images_file ( "C:\\Users\\H181523\\Downloads\\train-images.idx3-ubyte" , train_images );

	std::cout << "Done!" << std::endl;

	/*
	std::vector<HANDLE> thread;
	for ( std::size_t k(0)
		; k < NUM_THREADS
		; ++k
		)
	{
		thread . push_back ( CreateThread(NULL, 0, ThreadFunc, NULL, 0, NULL) );
	}
	for ( std::size_t k(0)
		; k < thread.size()
		; ++k
		)
	{
		if (thread[k]) 
		{
			// Optionally do stuff, such as wait on the thread.
		}
	}
	*/

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
	glutCreateWindow("red 3D lighted cube");
	glutDisplayFunc(display);
	glutIdleFunc(idle);
	init();
	glutMainLoop();
	return 0;
}
