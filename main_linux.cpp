#include <iostream>
#include <fstream>
#include <vector>
#include <GL/glut.h>
#include <math.h>
#include <boost/thread.hpp>
#include <GL/freeglut.h>
using namespace std;


struct Image
{
    std::vector < unsigned char > img_data;
};

std::vector < unsigned char > test_data;
std::vector < Image > test_img_data;
std::vector < unsigned char > train_data;
std::vector < Image > train_img_data;

struct Data
{
    std::vector < double > features;
    std::vector < double > labels;
};

std::vector < Data > examples;
std::vector < Data > test_examples;

struct Operator
{

    std::size_t m;
    std::size_t n;

    std::vector < std::vector < double > > W;

    Operator ( std::size_t p_m , std::size_t p_n )
        : m ( p_m )
        , n ( p_n )
    {
        for ( std::size_t i(0)
            ; i < m
            ; ++i
            )
        {
            std::vector < double > R;
            for ( std::size_t j(0)
                ; j < n
                ; ++j
                )
            {
                R . push_back ( 0 );
            }
            W . push_back ( R );
        }
    }

    void randomize ()
    {
        for ( std::size_t i(0)
            ; i < m
            ; ++i
            )
        {
            for ( std::size_t j(0)
                ; j < n
                ; ++j
                )
            {
                W[i][j] = 0.5 * ( 1.0 - 2.0 * ( ( rand () % 1000 ) / 1000.0 ) );
            }
        }
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

#if 0
struct Network
{

    std::vector < std::size_t > sizes;

    std::vector < std::size_t > bias_sizes;

    std::vector < Operator * > operators;

    void init ( std::vector < std::size_t > const & sizes 
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


    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                                                                                                                              //
    //  x[e] = [L_0 x 1]                                                                                                            //
    //                                                                                                                              //
    //  y[e] = [L_{M-1} x 1]                                                                                                        //
    //                                                                                                                              //
    //  a^l = [L_l x 1] , l in [0,M-1]                                                                                              //
    //                                                                                                                              //
    //  in^l = [L_l x 1] , l in [0,M-1]                                                                                             //
    //                                                                                                                              //
    //  W^l = [L_l x L_{l+1}] , l in [0,M-2]                                                                                        //
    //                                                                                                                              //
    //  /\^l = [L_l x 1] , l in [0,M-1]                                                                                             //
    //                                                                                                                              //
    //  repeat                                                                                                                      //
    //      for each e in examples do                                                                                               //
    //          for each node j in the input layer do a^0_j <- x_j[e]                                                               //
    //          for l=1 to M-1 do                                                                                                   //
    //              in^l_i <- sum_j W^{l-1}_j,i a^{l-1}_j                   // in^l = W^{l-1}^T a^{l-1}  ,  l in [1,M-1]            //
    //              a^l_i <- g(in^l_i)                                      // a^l = g(in^l)                                        //
    //          for each node i in the output layer do                                                                              //
    //              /\^l_i <- g'(in^{M-1}_i) x ( y_i[e] - a^{M-1}_i )       // /\^{M-1} = g'(in^{M-1}) ( y - a^{M-1} )              //
    //          for l = M - 2 to 0 do                                                                                               //
    //              for each node j in layer l do                           // l in [M-2,0]                                         //
    //                  /\^l_j <- g'(in^l_j) sum_i W^l_j,i /\^l_i           // /\^l = g'(in^l) W^l /\^{l+1}                         //
    //                  for each node i in layer l+1 do                                                                             //
    //                      W^l_j,i <- W^l_j,i + epsilon x a^l_j x /\^l_i   // W^l += epsilon a^l x /\^{l+1}                        //
    //  until convergence                                                                                                           //
    //                                                                                                                              //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    void NN_backpropagation ( std::vector < Data > const & examples )
    {
        {
            double error = 0;
            for ( std::size_t k(0)
                ; k < examples.size()
                ; ++k
                )
            {
                Data data = BM . forwardpropagation ( examples[k] );
                for ( std::size_t j(0)
                    ; j < data.features.size()
                    ; ++j
                    )
                {
                    a[0][j] = data.features[j];
                }
                for ( std::size_t l(1)
                    ; l < M
                    ; ++l
                    )
                {
                    for ( std::size_t i(0)
                        ; i < W[l-1][0].size()
                        ; ++i
                        )
                    {
                        in[l][i] = 0;
                        for ( std::size_t j(0)
                            ; j < W[l-1].size()
                            ; ++j
                            )
                        {
                            in[l][i] += W[l-1][j][i] * a[l-1][j];
                        }
                        a[l][i] = g ( in[l][i] );
                    }
                }
                for ( std::size_t i(0)
                    ; i < data.labels.size()
                    ; ++i
                    )
                {
                    D[M-1][i] = dg ( in[M-1][i] ) * ( data.labels[i] - a[M-1][i] );
                    error += pow( data.labels[i] - a[M-1][i] , 2 );
                }
                for ( std::size_t l = M-2
                    ; l < M
                    ; --l
                    )
                {
                    for ( std::size_t j(0)
                        ; j < W[l].size()
                        ; ++j
                        )
                    {
                        double sum = 0;
                        for ( std::size_t i(0)
                            ; i < W[l][j].size()
                            ; ++i
                            )
                        {
                            sum += W[l][j][i] * D[l+1][i];
                        }
                        D[l][j] = dg(in[l][j]) * sum;
                        for ( std::size_t i(0)
                            ; i < W[l][j].size()
                            ; ++i
                            )
                        {
                            W[l][j][i] += epsilon * a[l][j] * D[l+1][i];
                        }
                    }
                }
            }
            std::cout << "error=" << error << std::endl;
            return error;
        }
    }

    void optimize_Boltzmann_machine ()
    {

    }

    void optimize ()
    {
        
    }

};
#endif




class NeuralNetwork
{

    std::vector < std::vector < std::vector < double > > > W;
    std::vector < std::vector < std::vector < double > > > prev_W;
    std::vector < std::vector < std::vector < double > > > back_W;
    std::vector < std::vector < double > > in;
    std::vector < std::vector < double > > a;
    std::vector < std::vector < double > > D;
    std::size_t M;

public:

    void init( std::vector < std::size_t > const & sizes )
    {
        M = sizes . size ();
        for ( std::size_t k(0)
            ; k < sizes . size () - 1
            ; ++k
            )
        {
            std::vector < std::vector < double > > L;
            for ( std::size_t i(0)
                ; i < sizes[k]
                ; ++i
                )
            {
                std::vector < double > R;
                for ( std::size_t j(0)
                    ; j < sizes[k+1]
                    ; ++j
                    )
                {
                    R . push_back ( 1*(1 - 2 * (rand () % 1000) / 1000.0) );
                }
                L . push_back ( R );
            }
            W . push_back ( L );
        }
        for ( std::size_t k(0)
            ; k < sizes . size ()
            ; ++k
            )
        {
            in . push_back ( std::vector < double > ( sizes[k] , 0 ) );
            a  . push_back ( std::vector < double > ( sizes[k] , 0 ) );
            D  . push_back ( std::vector < double > ( sizes[k] , 0 ) );
        }
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                                                                                                                              //
    //  x[e] = [L_0 x 1]                                                                                                            //
    //                                                                                                                              //
    //  y[e] = [L_{M-1} x 1]                                                                                                        //
    //                                                                                                                              //
    //  a^l = [L_l x 1] , l in [0,M-1]                                                                                              //
    //                                                                                                                              //
    //  in^l = [L_l x 1] , l in [0,M-1]                                                                                             //
    //                                                                                                                              //
    //  W^l = [L_l x L_{l+1}] , l in [0,M-2]                                                                                        //
    //                                                                                                                              //
    //  /\^l = [L_l x 1] , l in [0,M-1]                                                                                             //
    //                                                                                                                              //
    //  repeat                                                                                                                      //
    //      for each e in examples do                                                                                               //
    //          for each node j in the input layer do a^0_j <- x_j[e]                                                               //
    //          for l=1 to M-1 do                                                                                                   //
    //              in^l_i <- sum_j W^{l-1}_j,i a^{l-1}_j                   // in^l = W^{l-1}^T a^{l-1}  ,  l in [1,M-1]            //
    //              a^l_i <- g(in^l_i)                                      // a^l = g(in^l)                                        //
    //          for each node i in the output layer do                                                                              //
    //              /\^l_i <- g'(in^{M-1}_i) x ( y_i[e] - a^{M-1}_i )       // /\^{M-1} = g'(in^{M-1}) ( y - a^{M-1} )              //
    //          for l = M - 2 to 0 do                                                                                               //
    //              for each node j in layer l do                           // l in [M-2,0]                                         //
    //                  /\^l_j <- g'(in^l_j) sum_i W^l_j,i /\^l_i           // /\^l = g'(in^l) W^l /\^{l+1}                         //
    //                  for each node i in layer l+1 do                                                                             //
    //                      W^l_j,i <- W^l_j,i + epsilon x a^l_j x /\^l_i   // W^l += epsilon a^l x /\^{l+1}                        //
    //  until convergence                                                                                                           //
    //                                                                                                                              //
    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    void forwardpropagation ( std::vector < Data > const & examples , std::vector < double > & prediction )
    {
        {
            for ( std::size_t k(0)
                ; k < examples.size()
                ; ++k
                )
            {
                Data data = examples[k];
                for ( std::size_t j(0)
                    ; j < data.features.size()
                    ; ++j
                    )
                {
                    a[0][j] = data.features[j];
                }
                for ( std::size_t l(1)
                    ; l < M
                    ; ++l
                    )
                {
                    for ( std::size_t i(0)
                        ; i < W[l-1][0].size()
                        ; ++i
                        )
                    {
                        in[l][i] = 0;
                        for ( std::size_t j(0)
                            ; j < W[l-1].size()
                            ; ++j
                            )
                        {
                            in[l][i] += W[l-1][j][i] * a[l-1][j];
                        }
                        a[l][i] = g ( in[l][i] );
                    }
                }
                double p = 0;
                for ( std::size_t i(0)
                    ; i < a[M-1].size()
                    ; ++i
                    )
                {
                    if ( a[M-1][i] > p )
                    {
                        p = a[M-1][i];
                        prediction[k] = i;
                    }
                }
            }
        }
    }
    void revert()
    {
        W = prev_W;
    }
    void save()
    {
        prev_W = W;
    }
    double backpropagation ( double epsilon , std::vector < Data > const & examples )
    {
        {
            double error = 0;
            for ( std::size_t k(0)
                ; k < examples.size()
                ; ++k
                )
            {
                //std::cout << "k=" << k << std::endl;
                Data data = examples[k];
                for ( std::size_t j(0)
                    ; j < data.features.size()
                    ; ++j
                    )
                {
                    a[0][j] = data.features[j];
                }
                for ( std::size_t l(1)
                    ; l < M
                    ; ++l
                    )
                {
                    for ( std::size_t i(0)
                        ; i < W[l-1][0].size()
                        ; ++i
                        )
                    {
                        in[l][i] = 0;
                        for ( std::size_t j(0)
                            ; j < W[l-1].size()
                            ; ++j
                            )
                        {
                            in[l][i] += W[l-1][j][i] * a[l-1][j];
                        }
                        a[l][i] = g ( in[l][i] );
                    }
                }
                for ( std::size_t i(0)
                    ; i < data.labels.size()
                    ; ++i
                    )
                {
                    D[M-1][i] = dg ( in[M-1][i] ) * ( data.labels[i] - a[M-1][i] );
                    error += pow( data.labels[i] - a[M-1][i] , 2 );
                }
                for ( std::size_t l = M-2
                    ; l < M
                    ; --l
                    )
                {
                    for ( std::size_t j(0)
                        ; j < W[l].size()
                        ; ++j
                        )
                    {
                        double sum = 0;
                        for ( std::size_t i(0)
                            ; i < W[l][j].size()
                            ; ++i
                            )
                        {
                            sum += W[l][j][i] * D[l+1][i];
                        }
                        D[l][j] = dg(in[l][j]) * sum;
                        for ( std::size_t i(0)
                            ; i < W[l][j].size()
                            ; ++i
                            )
                        {
                            W[l][j][i] += epsilon * a[l][j] * D[l+1][i];
                        }
                    }
                }
                //std::cout << "error=" << error << std::endl;
                //char ch;
                //std::cin >> ch;
            }
            //std::cout << "error=" << error << std::endl;
            return error;
        }
    }

    double g ( double x )
    {
        return log ( 1 + exp ( x ) );
    }

    double dg ( double x )
    {
        return 1.0 / ( 1.0 + exp ( -x ) );
    }

    // double g ( double x )
    // {
    //     return 1.0 / (1.0 + exp ( -x ));
    // }

    // double dg ( double x )
    // {
    //     return g(x) * ( 1 - g(x) );
    // }

};

NeuralNetwork NN;

boost::mutex mut;

std::size_t img_index = 0;

float img_index_flt = 0;

void RenderString(float x, float y, float z, void *font, const char* str)
{  
    glColor3f(1,1,1); 
    glRasterPos3f(x, y, z);
    glutBitmapString(font, str);
}

void
drawBox(void)
{
    img_index_flt += 0.06;
    img_index = (int)(img_index_flt) % test_img_data.size();
    double val;
    glBegin(GL_QUADS);
    if ( img_index < test_img_data . size () )
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
                val = (float)test_img_data[img_index].img_data[k]/256.0;
                glColor3f(val,val,val);
                glVertex3f( -1 + 2* y   /28.0 , -(-1 + 2* x   /28.0) , 0 );
                glVertex3f( -1 + 2* y   /28.0 , -(-1 + 2*(x+1)/28.0) , 0 );
                glVertex3f( -1 + 2*(y+1)/28.0 , -(-1 + 2*(x+1)/28.0) , 0 );
                glVertex3f( -1 + 2*(y+1)/28.0 , -(-1 + 2* x   /28.0) , 0 );
            }
        }
    }
    glEnd();
    std::vector < Data > t_test_examples;
    std::vector < double > prediction(1);
    {
        Data d;
        for ( std::size_t i(0)
            ; i < test_img_data[img_index].img_data.size()
            ; ++i
            )
        {
            d . features . push_back ( (double)test_img_data[img_index].img_data[i] / 256.0 );
        }
        d . features . push_back ( 1 );
        for ( std::size_t i(0)
            ; i < 10
            ; ++i
            )
        {
            if ( i == (int)test_data[img_index] )
            {
                d . labels . push_back ( 1 );
            }
            else
            {
                d . labels . push_back ( 0 );
            }
        }
        t_test_examples . push_back ( d );
    }
    mut . lock ();
    NN . forwardpropagation ( t_test_examples , prediction );
    mut . unlock ();
    std::stringstream ss;
    ss << (int)test_data[img_index] << "-" << prediction[0];
    RenderString ( 1 , 1 , 0.01 , GLUT_BITMAP_TIMES_ROMAN_24 , ss.str().c_str() );
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



int read_label_file ( std::string filename , std::vector < unsigned char > & data ) {
    streampos size;
    char * memblock;
    ifstream file (filename.c_str(), ios::in|ios::binary|ios::ate);
    if (file.is_open())
    {
        size = file.tellg();
        memblock = new char [size];
        file.seekg (0, ios::beg);
        file.read (memblock, size);
        file.close();
        cout << "the entire file content is in memory" << std::endl;
        unsigned short magic_number = (int)memblock[3]+(int)memblock[2]*256+(int)memblock[1]*256*256+(int)memblock[0]*256*256*256;
        unsigned short number_of_images = (int)memblock[7]+(int)memblock[6]*256+(int)memblock[5]*256*256+(int)memblock[4]*256*256*256;
        std::cout << "magic number:" << magic_number << std::endl;
        std::cout << "number of images:" << number_of_images << std::endl;
        for ( std::size_t k(0)
            ; k < number_of_images
            ; ++k
            )
        {
            data . push_back ( (unsigned char) (memblock[8+k]) );
            unsigned char ch ( memblock[8+k] );
        }
        delete[] memblock;
    }
    else cout << "Unable to open file";
    return 0;
}

int read_picture_file ( std::string filename , std::vector < Image > & data ) {
    streampos size;
    char * memblock;
    ifstream file (filename.c_str(), ios::in|ios::binary|ios::ate);
    if (file.is_open())
    {
        size = file.tellg();
        memblock = new char [size];
        file.seekg (0, ios::beg);
        file.read (memblock, size);
        file.close();
        cout << "the entire file content is in memory" << std::endl;
        unsigned short magic_number     = (int)memblock[3 ]+(int)memblock[2 ]*256+(int)memblock[1 ]*256*256+(int)memblock[0 ]*256*256*256;
        unsigned short number_of_images = (int)memblock[7 ]+(int)memblock[6 ]*256+(int)memblock[5 ]*256*256+(int)memblock[4 ]*256*256*256;
        unsigned short wx               = (int)memblock[11]+(int)memblock[10]*256+(int)memblock[9 ]*256*256+(int)memblock[8 ]*256*256*256;
        unsigned short wy               = (int)memblock[15]+(int)memblock[14]*256+(int)memblock[13]*256*256+(int)memblock[12]*256*256*256;
        std::cout << "magic number:" << magic_number << std::endl;
        std::cout << "number of images:" << number_of_images << std::endl;
        std::cout << "wx:" << wx << std::endl;
        std::cout << "wy:" << wy << std::endl;
        unsigned short num_pixels = wx * wy;
        for ( std::size_t img(0)
            , k(0)
            ; img < number_of_images
            ; ++img
            )
        {
            Image image;
            for ( std::size_t px(0)
                ; px < num_pixels
                ; ++px
                , ++k
                )
            {
                unsigned char ch ( memblock[16+k] );
                image . img_data . push_back ( ch );
            }
            data . push_back ( image );
            std::cout << img << std::endl;
        }
        std::cout << "done reading pictures" << std::endl;
        delete[] memblock;
    }
    else cout << "Unable to open file";
    return 0;
}

void train()
{
    double prev_error = 10000000;
    double error = 10000000;
    double init_epsilon = 1e-2;
    double epsilon = init_epsilon;
    double errors;

    std::vector < double > fwd_errors ( examples . size () );
    std::vector < double > test_fwd_errors ( test_examples . size () );

    for ( std::size_t iter(0); iter < 1000; iter++ )
    {
      std::cout << "pass 4" << std::endl;
      prev_error = error;
      mut . lock ();
      error = NN . backpropagation( epsilon , examples );
      mut . unlock ();
      std::cout << "pass 5" << std::endl;
      std::cout << "error=" << error << "\t   epsilon=" << epsilon << "\t";
      //if ( fabs ( error - prev_error ) / error < 0.0000001 )
      //if ( error < prev_error )
      //{
      //    epsilon *= 1.0001;
      //    if ( epsilon > 1 )
      //    {
      //        epsilon = 1;
      //    }
      //}
      //else
      //{
      //    epsilon = init_epsilon;

      //}
      //char ch;
      //std::cin >> ch;
      mut . lock ();
      NN . forwardpropagation ( examples , fwd_errors );
      mut . unlock ();
      double num_correct = 0;
      for ( std::size_t i(0)
          ; i < fwd_errors . size ()
          ; ++i
          )
      {
        if ( (int)train_data[i] == fwd_errors[i] )
        {
          num_correct += 1;
        }
      }
      mut . lock ();
      NN . forwardpropagation ( test_examples , test_fwd_errors );
      mut . unlock ();
      double test_num_correct = 0;
      for ( std::size_t i(0)
          ; i < test_fwd_errors . size ()
          ; ++i
          )
      {
        if ( (int)test_data[i] == test_fwd_errors[i] )
        {
          test_num_correct += 1;
        }
      }
      std::cout << "prediction rate = " << num_correct / fwd_errors . size () << "\t test prediction rate=" << test_num_correct / test_fwd_errors . size () << std::endl;
      sleep(10);
    }

}

int main(int argc,char ** argv)
{
    std::cout << "Hello to MNIST parser!" << std::endl;
    read_label_file ( "t10k-labels-idx1-ubyte" , test_data );
    read_picture_file ( "t10k-images-idx3-ubyte" , test_img_data );
    std::cout << test_img_data.size() << std::endl;

    /*
    for ( std::size_t k(0)
        ; k < test_img_data.size()
        ; ++k
        )
    {
        for ( std::size_t x(0)
            , i(0)
            ; x < 28
            ; ++x
            )
        {
            for ( std::size_t y(0)
                ; y < 28
                ; ++y
                , ++i
                )
            {
                std::cout << ((test_img_data[k].img_data[i]>127)?'*':' ');
            }
            std::cout << "\n";
        }
        std::cout << (int)test_data[k] << std::endl;
        char ch;
        std::cin >> ch;
    }
    */

    read_label_file ( "train-labels-idx1-ubyte" , train_data );
    read_picture_file ( "train-images-idx3-ubyte" , train_img_data );
    std::cout << train_img_data.size() << std::endl;

    /*
    for ( std::size_t k(0)
        ; k < train_img_data.size()
        ; ++k
        )
    {
        for ( std::size_t x(0)
            , i(0)
            ; x < 28
            ; ++x
            )
        {
            for ( std::size_t y(0)
                ; y < 28
                ; ++y
                , ++i
                )
            {
                std::cout << ((train_img_data[k].img_data[i]>127)?'*':' ');
            }
            std::cout << "\n";
        }
        std::cout << (int)train_data[k] << std::endl;
        char ch;
        std::cin >> ch;
    }
    */

    std::cout << "pass 1" << std::endl;

    for ( std::size_t k(0)
        ; k < train_data.size()
        ; ++k
        )
    {
        Data d;
        for ( std::size_t i(0)
            ; i < train_img_data[k].img_data.size()
            ; ++i
            )
        {
            d . features . push_back ( (double)train_img_data[k].img_data[i] / 256.0 );
        }
        d . features . push_back ( 1 );
        for ( std::size_t i(0)
            ; i < 10
            ; ++i
            )
        {
            if ( i == (int)train_data[k] )
            {
                d . labels . push_back ( 1 );
            }
            else
            {
                d . labels . push_back ( 0 );
            }
        }
        examples . push_back ( d );
    }

    for ( std::size_t k(0)
        ; k < test_data.size()
        ; ++k
        )
    {
        Data d;
        for ( std::size_t i(0)
            ; i < test_img_data[k].img_data.size()
            ; ++i
            )
        {
            d . features . push_back ( (double)test_img_data[k].img_data[i] / 256.0 );
        }
        d . features . push_back ( 1 );
        for ( std::size_t i(0)
            ; i < 10
            ; ++i
            )
        {
            if ( i == (int)test_data[k] )
            {
                d . labels . push_back ( 1 );
            }
            else
            {
                d . labels . push_back ( 0 );
            }
        }
        test_examples . push_back ( d );
    }

    std::cout << "pass 2" << std::endl;

    std::vector < std::size_t > sizes;

    sizes . push_back ( examples[0] . features . size () );
    //sizes . push_back ( 300 );
    sizes . push_back ( examples[0] . labels . size () );

    NN . init ( sizes );


    std::cout << "pass 3" << std::endl;

    boost::thread * th = new boost::thread ( train );

    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutCreateWindow("red 3D lighted cube");
    glutDisplayFunc(display);
    glutIdleFunc(idle);
    init();
    glutMainLoop();
    th -> join ();
    return 0;
}

