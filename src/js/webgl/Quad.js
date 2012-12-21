// Copyright 2002-2012, University of Colorado

/**
 * Renderable Quad (4-vertex rectangle)
 *
 * @author Jonathan Olson
 */

var phet = phet || {};
phet.webgl = phet.webgl || {};

// create a new scope
(function () {
    phet.webgl.Quad = function ( gl, width, height, xOffset, yOffset ) {
        phet.webgl.GLNode.call( this );

        this.gl = gl;
        this.width = width || 1;
        this.height = height || 1;
        this.xOffset = xOffset || 1;
        this.yOffset = yOffset || 1;

        this.setupBuffers();
    };

    var Quad = phet.webgl.Quad;

    Quad.prototype = Object.create( phet.webgl.GLNode.prototype );
    Quad.prototype.constructor = Quad;

    // TODO: should we allow buffer parameters to change?
    Quad.prototype.setupBuffers = function () {
        var gl = this.gl;
        // TODO: maybe not use a global reference to gl?
        this.positionBuffer = gl.createBuffer();
        this.normalBuffer = gl.createBuffer();
        this.indexBuffer = gl.createBuffer();
        this.textureBuffer = gl.createBuffer();

        var width = this.width;
        var height = this.height;
        var xOffset = this.xOffset;
        var yOffset = this.yOffset;

        var positionData = [
            xOffset, yOffset, 0,
            xOffset + width, yOffset, 0,
            xOffset + width, yOffset + height, 0,
            xOffset, yOffset + height, 0
        ];
        var normalData = [
            0, 0, 1,
            0, 0, 1,
            0, 0, 1,
            0, 0, 1
        ];

        // the two triangles
        var indexData = [
            0, 1, 2,
            0, 2, 3
        ];
        var textureData = [
            0, 0,
            1, 0,
            1, 1,
            0, 1
        ];

        console.log( normalData );
        console.log( textureData );
        console.log( positionData );
        console.log( indexData );

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

    Quad.prototype.renderSelf = function ( args ) {

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

        if( args.inverseTransposeAttribute !== null ) {
            gl.uniformMatrix4fv( args.inverseTransposeAttribute, false, args.transform.inverseTransposed.entries );
        }

        gl.bindBuffer( gl.ELEMENT_ARRAY_BUFFER, this.indexBuffer );
        gl.drawElements( gl.TRIANGLES, this.indexBuffer.numItems, gl.UNSIGNED_SHORT, 0 );
    };
})();
