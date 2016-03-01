/* C Librares */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
/* C++ Libraries */
#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <cstdlib>
#include <cmath>
#include <vector>

#include <GL/glut.h>
#include <GL/freeglut.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#define min(a,b) (((a) < (b)) ? (a) : (b))

#define WINDOW_TITTLE_PREFIX "Reaction-Diffusion"
#define GSZ 424


#define _CRT_SECURE_NO_WARNINGS
#pragma warning(disable:4996)

using namespace std;



int Cwidth = 1400, Cheight = 800;

int mouse_button = 1;

float r = 0.0f, g = 0.0f, b = 0.0f;
float T = 0.0f, dn;

unsigned Framecount = 0;

double u1[GSZ][GSZ], u2[GSZ][GSZ], u3[GSZ][GSZ], u1old[GSZ][GSZ], u2old[GSZ][GSZ], u3old[GSZ][GSZ];

int counter = 0;

int	tmax = 10000;                          // horizon in t
int nt = tmax * 110;                          // number of grid points in t
double delta_t = (double)tmax / nt;                 //  mesh-size in t

int nx = GSZ, ny = GSZ;    //  number of grid points in x  //  number of grid points in y

int Ls = 100;
int xmax = Ls, ymax = Ls;    //  horizon in x   //  horizon in y

double delta_x = (double)xmax / (nx + 1);           //  mesh-size in x
double delta_y = (double)ymax / (ny + 1);           //  mesh-size in x

double kappa = 0.025;

// Angles and zoom for the molecule.
double xangle = 15.0;
double yangle = 40.0;
double zoom = -260;
const double pi = acos(-1.0);

double xsmall = 0.0, ysmall = 0.0;
double xbig = Ls, ybig = Ls;
double xval[GSZ], yval[GSZ];

// Variables for old mouse coordinates.
int old_mx;
int old_my;


/* DEFINE FUNCTIONS */
void Initialize(void);

// Drawing functions GLUT will need.
void display(void);
void reshape(int, int);

// Input functions GLUT will need.
void keyboard(unsigned char, int, int);
void mouse_wheel(int, int, int, int);
void mouse_drag(int, int);
void mouse_move(int, int);
void draw_axes(void);

void render(void);
void FTCS(void);
void timer(int);
void idle(void);

void keyboard(unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);

void drawString(void *font, char *s);

bool wire = false;

void SaveTGA(int nShot);
int shot_num = 0;
int screenshot_mode = 0;

void SaveOxide(int oxide_num);
int oxide_num = 0;

void Initialize(void) {

    cout << "nx = " << nx << "  ny = " << ny << endl;


    // Fill in the x vals.
    for (int i = 0; i < nx; i++)
        xval[i] = xsmall + i*delta_x;

    // Fill in the y vals.
    for (int j = 0; j < nx; j++)
        yval[j] = ysmall + j*delta_y;

    double D = 1.0;                         //  diffusion coefficient
    double bxy = 4.0*D*delta_t / (delta_x*delta_x);  //   Stability parameter (b=<1); case where dx = dy

    cout << "Stability parameter = " << bxy << endl;

    //  initial conditions and boundary conditions
    double b0 = 1.0 / 4.0;
    double u0 = 2.0 / 3.0;
    double R = 2.0;
    double Rsqr = R*R;
    double cond = 0.0;

    int i0[] = { 50,100,300,125 };
    int j0[] = { 50,10,150,125 };

    for (int i = 0; i<nx; i++) {
        for (int j = 0; j<ny; j++) {
            u1[i][j] = b0;
            u2[i][j] = 0.0;
            u3[i][j] = 0.0;
        }
    }
    // define nucleation points
    for (int k = 0; k<(sizeof(i0) / sizeof(int)); k++) {
        int i0_ = i0[k];
        int j0_ = j0[k];
        for (int i = 0; i<nx; i++) {
            for (int j = 0; j<ny; j++) {
                cond = (double)(i - i0_)*(i - i0_) + (j - j0_)*(j - j0_);
                if (cond <= Rsqr) {
                    u1[i][j] = 0.0;
                    u2[i][j] = u0;
                }
            }
        }
    }

}

void FTCS(void) {

    memcpy(u1old, u1, sizeof(u1));
    memcpy(u2old, u2, sizeof(u2));
    memcpy(u3old, u3, sizeof(u3));

    //u1old = u1;
    //u2old = u2;
    //u3old = u3;
#pragma omp parallel for
    for (int i = 1; i<nx - 1; i++) {
        for (int j = 1; j<ny - 1; j++) {

            double u1var = u1old[i][j];
            double u2var = u2old[i][j];
            double u3var = u3old[i][j];

            u1[i][j] = delta_t*(-u2var*u1var) + u1var;
            u2[i][j] = delta_t*(-2.0*u2var*u1var + kappa*u3var +
                ((u2old[i + 1][j] - 2.0*u2var + u2old[i - 1][j]) / (delta_x*delta_x) +
                    (u2old[i][j + 1] - 2.0*u2var + u2old[i][j - 1]) / (delta_y*delta_y))) +
                u2var;
            u3[i][j] = delta_t*(3.0*u2var*u1var - kappa*u3var) + u3var;

        }
    }
}

void display(void) {
    //glClear(GL_COLOR_BUFFER_BIT);   // Clear Buffer
    ++Framecount;
    //glClearColor(0.f,0.f,0.f,0.f);  // Clear Background
    //glLoadIdentity();               // Open Identity Matrix

    render();                       // Do The Animation
}

void render(void) {
    //if (mouse_button == 1){
    FTCS();
    //}
    counter = counter + 1;
    if (counter % 1000 == 0) {
        float light_pos[] = { -zoom, -zoom, -zoom, 1.0 };
        float ambient[] = { 0.0, 0.0, 0.0, 1.0 };
        float diffuse[] = { 1.0, 1.0, 1.0, 1.0 };
        float specular[] = { 1.0, 1.0, 1.0, 1.0 };

        glClearColor(1.0, 1.0, 1.0, 1.0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glClearDepth(1);

        glCullFace(GL_BACK);

        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        glEnable(GL_LIGHT0);
        glEnable(GL_COLOR_MATERIAL);
        glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);

        glLightfv(GL_LIGHT0, GL_POSITION, light_pos);
        glLightfv(GL_LIGHT0, GL_AMBIENT, ambient);
        glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse);
        glLightfv(GL_LIGHT0, GL_SPECULAR, specular);

        glTranslatef(0.0, 0.0, zoom);
        glRotatef(xangle, 0, 1, 0);
        glRotatef(yangle, cos((pi / 180)*xangle), 0, sin((pi / 180)*xangle));

        glLineWidth(1.0);

        // glEnable(GL_AUTO_NORMAL);
        glDisable(GL_LIGHTING);

        glBegin(GL_TRIANGLES);
        double scale = 40.0;
        int pos = Ls / 2;

        //glOrtho(-2, nx+2, -2, ny+2,-1,1); // Multiplies the current matrix by an orthographic matrix. (left, right, bottom, top, near, far);
        for (int i = 1; i < nx - 2; i++) {
            for (int j = 1; j < ny - 2; j++) {
                // Returns the appropriate value from the Jet color function.
                float fourValue = 11 * u3[i][j];
                float red = min(fourValue - 1.5, -fourValue + 4.5);
                float green = min(fourValue - 0.5, -fourValue + 3.5);
                float blue = min(fourValue + 0.5, -fourValue + 2.5);
                glColor3f(red, green, blue);
                //glColor3f(0.2 + ((u3[i][j]/1)*0.9), 0.2, 0.2);
                // Draw the first face. X Z Y
                glVertex3d(xval[i] - (xbig - xsmall) / 2 - pos,
                    u3[i][j] * scale,
                    (ybig - yval[j]) - (ybig - ysmall) / 2);

                glVertex3d(xval[i + 1] - (xbig - xsmall) / 2 - pos,
                    u3[i + 1][j] * scale,
                    (ybig - yval[j]) - (ybig - ysmall) / 2);

                glVertex3d(xval[i + 1] - (xbig - xsmall) / 2 - pos,
                    u3[i + 1][j + 1] * scale,
                    (ybig - yval[j + 1]) - (ybig - ysmall) / 2);

                // Draw the second face. X Z Y
                glVertex3d(xval[i] - (xbig - xsmall) / 2 - pos,
                    u3[i][j] * scale,
                    (ybig - yval[j]) - (ybig - ysmall) / 2);

                glVertex3d(xval[i + 1] - (xbig - xsmall) / 2 - pos,
                    u3[i + 1][j + 1] * scale,
                    (ybig - yval[j + 1]) - (ybig - ysmall) / 2);

                glVertex3d(xval[i] - (xbig - xsmall) / 2 - pos,
                    u3[i][j + 1] * scale,
                    (ybig - yval[j + 1]) - (ybig - ysmall) / 2);

                // Returns the appropriate value from the Jet color function.
                fourValue = 8 * u1[i][j];
                red = min(fourValue - 1.5, -fourValue + 4.5);
                green = min(fourValue - 0.5, -fourValue + 3.5);
                blue = min(fourValue + 0.5, -fourValue + 2.5);
                glColor3f(red, green, blue);
                //glColor3f(0.2 + ((u3[i][j]/1)*0.9), 0.2, 0.2);
                // Draw the first face. X Z Y
                glVertex3d(xval[i] - (xbig - xsmall) / 2 + pos,
                    u1[i][j] * scale,
                    (ybig - yval[j]) - (ybig - ysmall) / 2);

                glVertex3d(xval[i + 1] - (xbig - xsmall) / 2 + pos,
                    u1[i + 1][j] * scale,
                    (ybig - yval[j]) - (ybig - ysmall) / 2);

                glVertex3d(xval[i + 1] - (xbig - xsmall) / 2 + pos,
                    u1[i + 1][j + 1] * scale,
                    (ybig - yval[j + 1]) - (ybig - ysmall) / 2);

                // Draw the second face. X Z Y
                glVertex3d(xval[i] - (xbig - xsmall) / 2 + pos,
                    u1[i][j] * scale,
                    (ybig - yval[j]) - (ybig - ysmall) / 2);

                glVertex3d(xval[i + 1] - (xbig - xsmall) / 2 + pos,
                    u1[i + 1][j + 1] * scale,
                    (ybig - yval[j + 1]) - (ybig - ysmall) / 2);

                glVertex3d(xval[i] - (xbig - xsmall) / 2 + pos,
                    u1[i][j + 1] * scale,
                    (ybig - yval[j + 1]) - (ybig - ysmall) / 2);

            }
        }
        glEnd();
        //draw_axes();
        //glTranslatef(-20, -20, -30);
        glRotatef(xangle, 0, 1, 0);
        glRotatef(yangle, cos((pi / 180)*xangle), 0, sin((pi / 180)*xangle));
        //glLoadIdentity();
        glutSwapBuffers();
        //glutPostRedisplay();
        if (screenshot_mode) {
            SaveTGA(shot_num);
            shot_num++;
        }
        SaveOxide(oxide_num);
        oxide_num++;
        cout << counter << endl;
    }
}


void idle(void) {
    glutPostRedisplay();
}

void timer(int Value) {
    if (0 != Value) {
        char* TempString = (char*)malloc(1024 + strlen(WINDOW_TITTLE_PREFIX));
        sprintf(
            TempString,
            "%s - %d Frames Per Second at %d x %d",
            WINDOW_TITTLE_PREFIX,
            Framecount * 4,
            Cwidth,
            Cheight
            );
        glutSetWindowTitle(TempString);
        free(TempString);
    }
    Framecount = 0;
    glutTimerFunc(250, timer, 1);
}

void keyboard(unsigned char key, int x, int y) {

    switch (key) {
    case 'w':
        glutReshapeWindow(800, 600);
        glutPositionWindow(100, 100);
        break;
    case 'f':
        glutFullScreen();
        break;
    case 'a':
        if (wire == true) {
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
            wire = false;
        }
        else {
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
            wire = true;
        }
        break;
    case 'c':
        if (screenshot_mode)
            screenshot_mode = 0;
        else
            screenshot_mode = 1;
        break;
    case 'q':
        exit(0); //Exit the program
        break;
    case 27: //Escape key
        exit(0); //Exit the program
        break;
    default:
        break;
    }
}

void mouse(int button, int state, int x, int y) {
    switch (button) {
    case GLUT_LEFT_BUTTON:
        mouse_button = 1;
        break;
    case GLUT_RIGHT_BUTTON:
        mouse_button = 0;
        break;
    }
}


/* MAIN FUNCTION */
int main(int argc, char** argv) {


#ifdef _OPENMP
    // int max_threads; //, nprocs, tid, num_threads;
    //nprocs = omp_get_num_procs();
    //max_threads = omp_get_max_threads();
    //cout << "Max OpenMP Threads = " << max_threads << endl;
    omp_set_num_threads(4);

    // #else /* _OPENMP */
    //   nprocs = 1;
    //   max_threads = 1;
#endif /* _OPENMP */

    Initialize();

    glutInit(&argc, argv);                          // Initialize GLUT

    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);   // Double RGB Buffer
    glutInitWindowSize(Cwidth, Cheight);			//Set the window size
    glutInitWindowPosition(100, 10);

    glutCreateWindow(WINDOW_TITTLE_PREFIX);

    //glShadeModel(GL_FLAT);

    glutDisplayFunc(display);
    glutIdleFunc(display);
    //glutIdleFunc(idle);
    glutTimerFunc(0, timer, 0);

    glutReshapeFunc(reshape);
    glutKeyboardFunc(keyboard);
    glutMouseFunc(mouse_wheel);
    glutMotionFunc(mouse_drag);
    glutPassiveMotionFunc(mouse_move);
    //glutMouseFunc(mouse);

    //glClear(GL_COLOR_BUFFER_BIT);   // Clear Buffer

    //glClearColor(0.f,0.f,0.f,0.f);  // Clear Background
    // glLoadIdentity();

    glutMainLoop();         // Main Loop of OpenGL

    return 0;

}


void mouse_wheel(int button, int state, int x, int y)
{
    if (state == GLUT_UP)
    {
        if (button == 3) // Mouse wheel up.
        {
            zoom -= 5;
        }
        else if (button == 4) // Mouse wheel down.
        {
            zoom += 5;
        }
    }
}

void mouse_drag(int x, int y)
{
    xangle += -(old_mx - x)*0.02;
    yangle += -(old_my - y)*0.02;

}

void mouse_move(int x, int y)
{
    old_mx = x;
    old_my = y;
}


void reshape(int w, int h)
{
    glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();


    glLineWidth(5.0);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glEnable(GL_LIGHTING);

    glShadeModel(GL_SMOOTH);

    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    gluPerspective(30.0f, (double)w / (double)h, 2.0, 5000.0);
    //glOrtho(-30, 30, -30, 30, 20, 2000.0);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}


void draw_axes()
{

    glLineWidth(2.0);
    double len = 9.0;
    int pos = Ls / 2 + 2;

    glColor3f(0.0, 0.0, 0.0);
    glBegin(GL_LINES);

    glColor3f(1.0, 0.0, 0.0);
    glVertex3d(pos, 0, pos);
    glVertex3d(len + pos, 0, pos);

    glColor3f(0.0, 1.0, 0.0);
    glVertex3d(pos, 0, pos);
    glVertex3d(pos, len, pos);

    glColor3f(0.0, 0.0, 1.0);
    glVertex3d(pos, 0, pos);
    glVertex3d(pos, 0, len + pos);
    glEnd();

    glRasterPos3d(len + pos, 0, pos);
    drawString(GLUT_BITMAP_HELVETICA_18, " X axis");
    glRasterPos3d(pos, len, pos);
    drawString(GLUT_BITMAP_HELVETICA_18, " Y axis");
    glRasterPos3d(pos, 0, len + pos);
    drawString(GLUT_BITMAP_HELVETICA_18, " Z axis");

}


void drawString(void *font, char *s)
{
    int i;

    //glDisable(GL_TEXTURE_2D);
    for (i = 0; i < strlen(s); i++) {
        glutBitmapCharacter(font, s[i]);
    }
    //glEnable(GL_TEXTURE_2D);
}


void SaveTGA(int nShot)
{

    glPixelStorei(GL_PACK_ALIGNMENT, 1);

    char cFileName[64];
    FILE *fScreenshot;
    int nSize = Cwidth * Cheight * 3;

    GLubyte *pixels = new GLubyte[nSize];
    if (pixels == NULL)
        return;

    sprintf(cFileName, "frame_%06d.tga", nShot);

    fScreenshot = fopen(cFileName, "wb");


    glReadPixels(0, 0, Cwidth, Cheight, GL_RGB, GL_UNSIGNED_BYTE, pixels);
    //convert to BGR format
    unsigned char temp;
    int i = 0;
    while (i < nSize)
    {
        temp = pixels[i];       //grab blue
        pixels[i] = pixels[i + 2];//assign red to blue
        pixels[i + 2] = temp;     //assign blue to red

        i += 3;     //skip to next blue byte
    }

    //*/

    //glReadPixels(0, 0, 512, 512, GL_BGR, GL_UNSIGNED_BYTE, pixels);

    unsigned char TGAheader[12] = { 0,0,2,0,0,0,0,0,0,0,0,0 };
    unsigned char header[6] = { Cwidth % 256, Cwidth / 256, Cheight % 256, Cheight / 256, 24, 0 };

    fwrite(TGAheader, sizeof(unsigned char), 12, fScreenshot);
    fwrite(header, sizeof(unsigned char), 6, fScreenshot);
    fwrite(pixels, sizeof(GLubyte), nSize, fScreenshot);
    fclose(fScreenshot);

    delete[] pixels;

    return;
}



void SaveOxide(int oxide_num)
{

    char cFileName[64];
    FILE *fp_Oxide;

    sprintf(cFileName, "oxide.txt");

    if (oxide_num == 0) {
        fp_Oxide = fopen(cFileName, "w");
    }
    else {
        fp_Oxide = fopen(cFileName, "a");
    }

    double sum = 0.0;
    // TODO fix nx-1, ny-1
    for (int i = 1; i<nx - 1; i++) {
        for (int j = 1; j<ny - 1; j++) {
            if (u1[i][j] < 0.05) {
                sum += 1.0;
            }
        }
    }

    sum = (double)sum / ((nx - 1)*(ny - 1));


    fprintf(fp_Oxide, "%0.3f %0.3f\n", counter*delta_t, sum);

    fclose(fp_Oxide);


    return;
}




/*

void resize_matrix(vector< vector<double*> > u){
u.resize(nx);
for (int i=0; i<nx; i++) {
u[i].resize(ny);
}
}



memcpy(u1old,u1,sizeof(u1));


int **array;
array = malloc(nrows * sizeof(int *));
if(array == NULL)
{
fprintf(stderr, "out of memory\n");
exit or return
}
for(i = 0; i < nrows; i++)
{
array[i] = malloc(ncolumns * sizeof(int));
if(array[i] == NULL)
{
fprintf(stderr, "out of memory\n");
exit or return
}
}



void zeroit(int **array, int nrows, int ncolumns)
{
int i, j;
for(i = 0; i < nrows; i++)
{
for(j = 0; j < ncolumns; j++)
array[i][j] = 0;
}
}


for(i = 0; i < nrows; i++)
free(array[i]);
free(array);






*/
