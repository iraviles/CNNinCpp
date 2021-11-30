#pragma once

#include<Eigen/Core>
#include<vector>
#include "Config.h"
#include "RNG.h"
#include"Optimizer.h"

class Layer{

protected:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;

    const int m_in_size;  //Layer's input size
    const int m_out_size; //Layer's output size

public:
    Layer(const in_size, const out_size):
        m_in_size(in_size),
        mout_size(out_size){}
    virtual ~Layer(); //Ser� heredada por otros tipos de capas (polimorfismo)

    int in_size() const{return m_in_size;}
    int out_size() const{return m_out_size;}

    virtual void init (const Scalar & mu, const Scalar & sigma, RNG & rng ) = 0;
    virtual void forward(const Matrix & prev_layer_output) = 0;

    virtual const Matrix & output const = 0;

    virtual void backprop(const Matrix & prev_layer_output, const Matrix & next_layer_data) = 0;
    virtual const Matrix & backprop_data() const = 0;

    virtual std::vector<Scalar> get_parameter() const = 0;
    virtual void set_parameters (const std::vector<Scalar> & param){}
    virtual std::vector<Scalar> get_derivatives() const = 0;
};
