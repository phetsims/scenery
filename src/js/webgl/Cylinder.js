// Copyright 2002-2012, University of Colorado

/**
 * Renderable Cylinder
 *
 * @author Jonathan Olson
 */

var phet = phet || {};
phet.webgl = phet.webgl || {};

// create a new scope
(function () {
    phet.webgl.Cylinder = function ( gl, radius, length, radialStrips, axisStrips ) {
        phet.webgl.GLNode.call( this );

        this.gl = gl;
        this.radius = radius;
        this.length = length;
        this.radialStrips = radialStrips || 16;
        this.axisStrips = axisStrips || 1;

        this.setupBuffers();
    };

    var Cylinder = phet.webgl.Cylinder;

    Cylinder.prototype = Object.create( phet.webgl.GLNode.prototype );
    Cylinder.prototype.constructor = Cylinder;

    // TODO: should we allow buffer parameters to change?
    Cylinder.prototype.setupBuffers = function () {
        var gl = this.gl;

        // TODO: maybe not use a global reference to gl?
        this.positionBuffer = gl.createBuffer();
        this.normalBuffer = gl.createBuffer();
        this.indexBuffer = gl.createBuffer();
        this.textureBuffer = gl.createBuffer();

        var positionData = [];
        var normalData = [];
        var indexData = [];
        var textureData = [];

        var radialSamples = this.radialStrips + 1;
        var axisSamples = this.axisStrips + 1;


        for ( var thetaIndex = 0; thetaIndex < radialSamples; thetaIndex++ ) {
            var u = thetaIndex / (radialSamples - 1);
            var phi = u * 2 * Math.PI;

            var x = Math.cos( phi );
            var y = Math.sin( phi );

            for ( var axisIndex = 0; axisIndex < axisSamples; axisIndex++ ) {
                var v = axisIndex / (axisSamples - 1);

                // center at origin
                var z = (v - 0.5) * this.length;

                positionData.push( x * this.radius );
                positionData.push( y * this.radius );
                positionData.push( z );

                normalData.push( x );
                normalData.push( y );
                normalData.push( 0 );

                textureData.push( u );
                textureData.push( v );

                if ( thetaIndex > 0 && axisIndex > 0 ) {
                    var baseA = (thetaIndex - 1) * axisSamples;
                    var baseB = thetaIndex * axisSamples;

                    // upper-left triangle
                    indexData.push( baseA + axisIndex - 1 );
                    indexData.push( baseB + axisIndex - 1 );
                    indexData.push( baseB + axisIndex );

                    // lower-right triangle
                    indexData.push( baseA + axisIndex - 1 );
                    indexData.push( baseB + axisIndex );
                    indexData.push( baseA + axisIndex );
                }
            }
        }

        gl.bindBuffer( gl.ARRAY_BUFFER, this.normalBuffer );
        gl.bufferData( gl.ARRAY_BUFFER, new Float32Array( normalData ), gl.STATIC_DRAW );
        this.normalBuffer.itemSize = 3;
        this.normalBuffer.numItems = normalData.length / 3;

        gl.bindBuffer( gl.ARRAY_BUFFER, this.textureBuffer );
        gl.bufferData( gl.ARRAY_BUFFER, new Float32Array( textureData ), gl.STATIC_DRAW );
        this.textureBuffer.itemSize = 2;
        this.textureBuffer.numItems = textureData.length / 2;

        gl.bindBuffer( gl.ARRAY_BUFFER, this.positionBuffer );
        gl.bufferData( gl.ARRAY_BUFFER, new Float32Array( positionData ), gl.STATIC_DRAW );
        this.positionBuffer.itemSize = 3;
        this.positionBuffer.numItems = positionData.length / 3;

        gl.bindBuffer( gl.ELEMENT_ARRAY_BUFFER, this.indexBuffer );
        gl.bufferData( gl.ELEMENT_ARRAY_BUFFER, new Uint16Array( indexData ), gl.STREAM_DRAW );
        this.indexBuffer.itemSize = 3;
        this.indexBuffer.numItems = indexData.length;
    };

    Cylinder.prototype.renderSelf = function ( args ) {

        var gl = this.gl;

        gl.bindBuffer( gl.ARRAY_BUFFER, this.positionBuffer );
        gl.vertexAttribPointer( args.positionAttribute, this.positionBuffer.itemSize, gl.FLOAT, false, 0, 0 );

        if ( args.textureCoordinateAttribute !== null ) {
            gl.bindBuffer( gl.ARRAY_BUFFER, this.textureBuffer );
            gl.vertexAttribPointer( args.textureCoordinateAttribute, this.textureBuffer.itemSize, gl.FLOAT, false, 0, 0 );
        }

        if ( args.normalAttribute !== null ) {
            gl.bindBuffer( gl.ARRAY_BUFFER, this.normalBuffer );
            gl.vertexAttribPointer( args.normalAttribute, this.normalBuffer.itemSize, gl.FLOAT, false, 0, 0 );
        }

        if ( args.transformAttribute !== null ) {
            gl.uniformMatrix4fv( args.transformAttribute, false, args.transform.matrix.entries );
        }

        if ( args.inverseTransposeAttribute !== null ) {
            gl.uniformMatrix4fv( args.inverseTransposeAttribute, false, args.transform.inverseTransposed.entries );
        }

        gl.bindBuffer( gl.ELEMENT_ARRAY_BUFFER, this.indexBuffer );
        gl.drawElements( gl.TRIANGLES, this.indexBuffer.numItems, gl.UNSIGNED_SHORT, 0 );
    };
})();
