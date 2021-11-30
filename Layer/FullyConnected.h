#pragma once

#include<Eigen/Core>
#include<vector>
#include<stdexcept>
#include"../Config.h"
#include"../Layer.h"
#include"../Utils/Random.h"

template<typename Activation>
class FullyConnected : public Layer{

private:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;

    Matrix m_weight;
    Vector m_bias;
    Matrix m_dw;    //Derivative of  weight matrix
    Vector m_db;    //Derivative of bias
    Matrix m_z;     //weight * input + bias
    Matrix m_a;     //Applying of activation function to z
    Matrix m_din;   //Derivative of input

public:
    FullyConnected(const int in_size, const int out_size):
        Layer(in_size, out_size){
        }

    void init(const Scalar & mu, const Scalar & sigma, RNG & rng){
        //Se inicializará la la matriz de pesos y el vector bias usando la función
        // de distribución normal, con media mu y desviación estándar sigma.

        m_weight.resize(this->m_in_size, this->m_out_size);
        m_bias.resize(this->m_out_size);
        m_dw.resize(this->m_in_size, this->m_out_size);
        m_db.resize(this->m_out_size);

        internal::set_normal_random(m_weight.data(), m_weight.size(), rng, mu, sigma);
        internal::set_normal_random(m_bias.data(), m_bias.size(), rng, mu, sigma);
    }

    void forward(const Matrix & prev_layer_data){
        const int nobs = prev_layer_data.col();

        // z = w' * input + bias
        m_z.resize(this->m_out_size, nobs);
        m_z.noalias() = m_weight.transpose() * prev_layer_data;
        w_z.colwise() += bias;

        //Applying activation function
        m_a.resize(this->m_out_size, nobs);
        Ativation::activate(m_z, m_a);
    }

    const Matrix & output() const{

    return m_a;
    }

    void backprop(const Matrix & prev_layer_data, const Matrix & next_layer_data){
        //TODO
    }

    const Matrix & backprop_data() const{
        return m_din;
    }

    void update(Optimizer & opt){
        ConstAlignedMapVec dw(m_dw.data(), m_dw.size());
        ConstAlignedMapVec db(m_db.data(), m_db.size());
        AlignedMapVec w(m_weight.data(), m_weight.size());
        AlignedMapVec d(m_bias.data(), m_bias.size());

        opt.update(dw, w);
        opt.update(db, b);
    }

    std::vector<Scalar> get_parameter() const {}
    void set_parameters(const std::vector<Scalar> & param){}
    std::vector<Scalar> get_derivatives() const{}

};
