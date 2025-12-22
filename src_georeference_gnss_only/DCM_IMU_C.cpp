#include "DCM_IMU_C.h"


DCM_IMU_C::DCM_IMU_C(const double Gravity, const double *State, const double *Covariance,
		const double DCMVariance, //const double BiasVariance,
		const double InitialDCMVariance, //const double InitialBiasVariance,
		const double MeasurementVariance, const double MeasurementVarianceVariableGain) :
		g0(Gravity), q_dcm2(DCMVariance), //q_gyro_bias2(BiasVariance),
		r_acc2(MeasurementVariance), r_a2(MeasurementVarianceVariableGain) 
{

	if (State == NULL) {
		double temp[] = DEFAULT_state;
		std::cout<<"Using DEFAULT_state"<<std::endl;
		for (int i = 0; i < s_size; ++i) {
			x[i] = temp[i];
			std::cout<<" "<<temp[i];
		}
		
	}
	else {
		std::cout<<"State is not NULL"<<std::endl;
		for (int i = 0; i < s_size; ++i) {
			x[i] = State[i];
			std::cout<<" "<<State[i];
		}
	}
	std::cout<<""<<std::endl;

	P.setZero();
	if (Covariance == NULL) {
		P(0,0) = InitialDCMVariance;
		P(1,1) = InitialDCMVariance;
		P(2,2) = InitialDCMVariance;
		// P(3,3) = InitialBiasVariance;
		// P(4,4) = InitialBiasVariance;
		// P(5,5) = InitialBiasVariance;
	}
	else {
		for (int i = 0; i < s_size; ++i) {
			for (int j = 0; j < s_size; ++j) {
				P(i,j) = Covariance[s_size*i + j];
			}
		}
	}

	H.setZero();
	H(0,0) = g0;
	H(1,1) = g0;
	H(2,2) = g0;

	Q.setZero();
	Q(0,0) = q_dcm2;
	Q(1,1) = q_dcm2;
	Q(2,2) = q_dcm2;
	// Q(3,3) = q_gyro_bias2;
	// Q(4,4) = q_gyro_bias2;
	// Q(5,5) = q_gyro_bias2;

	first_row << 1.0, 0.0, 0.0;   //what is this ?
	yaw = 0;
	pitch = 0;
	roll = 0;
}

void DCM_IMU_C::updateIMU(const Eigen::Vector3d &Gyroscope, const Eigen::Vector3d &Accelerometer, const double SamplePeriod)
{
	Eigen::Matrix<double, s_size, s_size>  Q_ = SamplePeriod*SamplePeriod * Q; // Process noise covariance with time dependent noise

	Eigen::Matrix<double, 3, 1> u = Gyroscope; // Control input (angular velocities from gyroscopes)

	// "rotation operators"
	Eigen::Matrix<double, 3, 3> C3X;
	C3X <<  0,    -x[2],  x[1],
			x[2],  0,   -x[0],
			-x[1], x[0], 0;

	//hat(w - b_w)

	Eigen::Matrix<double, 3, 3> UX; //bias free gyroscope velocity to skew symmetric form 
	// UX << 0,         -u[2]+x[5],  u[1]-x[4],
	// 		u[2]-x[5],  0,         -u[0]+x[3],
	// 		-u[1]+x[4],  u[0]-x[3],  0;
	// hat(w)
	UX <<  0,     -u[2],    u[1],
		  u[2],     0,     -u[0],
		  -u[1],   u[0],     0;


	// Model generation
	Eigen::Matrix<double, s_size, s_size> A;
	A.setZero();
	A.block<3,3>(0,3) = -SamplePeriod*C3X;

	Eigen::Matrix<double, s_size, 3> B;
	B.setZero();
	B.block<3,3>(0,0) = SamplePeriod*C3X;

	Eigen::Matrix<double, s_size, s_size> F;
	F.setZero();
	F.block<3,3>(0,0) = -SamplePeriod*UX;
	// F.block<3,3>(0,3) = -SamplePeriod*C3X;
	F += Eigen::Matrix<double, s_size, s_size>::Identity();


	// Kalman a priori prediction
	Eigen::Matrix<double, s_size, 1> x_predict;
	x_predict = x + A*x + B*u;

	Eigen::Matrix<double, s_size, s_size> P_predict;
	P_predict = F * P * F.transpose() + Q_;


	// measurements/observations (acceleromeres)
	Eigen::Matrix<double, 3, 1> z = Accelerometer;

	// recompute R using the error between acceleration and the model of g
	// (estimate of the magnitude of a0 in a = a0 + g)
	Eigen::Matrix<double, 3, 1> a_predict;
	a_predict = z - x_predict.block<3,1>(0,0)*g0;
	double a_len = sqrt(a_predict[0]*a_predict[0] + a_predict[1]*a_predict[1] + a_predict[2]*a_predict[2]);

	Eigen::Matrix<double, 3, 3> R;
	R = (a_len*r_a2 + r_acc2) * Eigen::Matrix<double, 3, 3>::Identity();


	// Kalman innovation
	Eigen::Matrix<double, 3, 1> y;
	y = z - H*x_predict;

	Eigen::Matrix<double, 3, 3> S;
	S = H * P_predict * H.transpose() + R;


	// Kalman gain
	Eigen::Matrix<double,s_size, 3> K;
	K = P_predict * H.transpose() * S.inverse();

	//save previous x to memory
	Eigen::Matrix<double, s_size, 1> x_last = x;

	// update a posteriori
	x = x_predict + K * y;

	// update a posteriori covariance
	Eigen::Matrix<double, s_size, s_size> IKH;
	IKH = Eigen::Matrix<double, s_size, s_size>::Identity() - K*H;
	P = IKH * P_predict * IKH.transpose() + K * R * K.transpose();

	// normalization of x & P (divide by DCM vector length)
	double d = sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]);

	Eigen::Matrix<double, s_size, s_size> J;
	J = Eigen::Matrix<double, s_size, s_size>::Identity();
	J.block<3,3>(0,0) << x[1]*x[1] + x[2]*x[2], -x[0]*x[1],             -x[0]*x[2],
						-x[0]*x[1],           	 x[0]*x[0] + x[2]*x[2], -x[1]*x[2],
						-x[0]*x[2],             -x[1]*x[2],              x[0]*x[0] + x[1]*x[1];
	J.block<3,3>(0,0) /= (d*d*d);

	// Laplace approximation of normalization function for x to P, J = Jacobian(f,x)
	P = J* P *J.transpose();
	x.block<3,1>(0,0) /= d;


	// compute Euler angles (not exactly a part of the extended Kalman filter)
	// yaw integration through full rotation matrix
	Eigen::Matrix<double, 3, 1> u_nb;
	u_nb = u - x.block<3,1>(3,0);

	if (true) {
		double cy = cos(yaw); //old angles (last state before integration)
		double sy = sin(yaw);
		double d = sqrt(x_last[1]*x_last[1] + x_last[2]*x_last[2]);
		double d_inv = 1.0 / d;

		// compute needed parts of rotation matrix R (state and angle based version, equivalent with the commented version above)
		double R11 = cy * d;
		double R12 = -(x_last[2]*sy + x_last[0]*x_last[1]*cy) * d_inv;
		double R13 = (x_last[1]*sy - x_last[0]*x_last[2]*cy) * d_inv;
		double R21 = sy * d;
		double R22 = (x_last[2]*cy - x_last[0]*x_last[1]*sy) * d_inv;
		double R23 = -(x_last[1]*cy + x_last[0]*x_last[2]*sy) * d_inv;

		// update needed parts of R for yaw computation
		double R11_new = R11 + SamplePeriod*(u_nb[2]*R12 - u_nb[1]*R13);
		double R21_new = R21 + SamplePeriod*(u_nb[2]*R22 - u_nb[1]*R23);
		yaw = atan2(R21_new,R11_new);
	}
	else { //alternative method estimating the whole rotation matrix
		//integrate full rotation matrix (using first row estimate in memory)
		Eigen::Matrix<double, 3, 1> x1, x2;
		x1 = first_row + SamplePeriod * UX.transpose() * first_row; //rotate x1 by x1 x u_nb
		x2 = C3X * x1; //second row x2 = (state x x1)
		x2 /= sqrt(x2[0]*x2[0] + x2[1]*x2[1] + x2[2]*x2[2]); // normalize length of the second row
		x1 = C3X.transpose() * x2; // recalculate first row x1 = (x2 * state) (ensure perpendicularity)
		first_row = x1 / sqrt(x1[0]*x1[0] + x1[1]*x1[1] + x1[2]*x1[2]); // normalize length
		yaw = atan2(x2[0],first_row[0]);
	}

	// compute new pitch and roll angles from a posteriori states
	pitch = asin(-x[0]);
	roll = atan2(x[1],x[2]);

	// save the estimated non-gravitational acceleration
	a = z - x.block<3,1>(0,0)*g0;
}

void DCM_IMU_C::getState(double *State) {
	for (int i = 0; i < s_size; ++i) {
		State[i] = x[i];
	}
}

void DCM_IMU_C::getCovariance(double *Covariance) {
	for (int i = 0; i < s_size; ++i) {
		for (int j = 0; j < s_size; ++j) {
			Covariance[s_size*i + j] = P(i,j);
		}
	}
}

void DCM_IMU_C::getNGAcc(double *ngacc) {
	for (int i = 0; i < 3; ++i) {
		ngacc[i] = a[i];
	}
}

