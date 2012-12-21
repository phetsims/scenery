// Copyright 2002-2012, University of Colorado

/**
 * Renderable Sphere
 *
 * @author Jonathan Olson
 */

var phet = phet || {};
phet.webgl = phet.webgl || {};

// create a new scope
(function () {
    phet.webgl.Sphere = function ( gl, radius, latitudeStrips, longitudeStrips ) {
        phet.webgl.GLNode.call( this );

        this.gl = gl;
        this.radius = radius;
        this.latitudeStrips = latitudeStrips || 10;
        this.longitudeStrips = longitudeStrips || 10;

        this.setupBuffers();
    };

    var Sphere = phet.webgl.Sphere;

    Sphere.prototype = Object.create( phet.webgl.GLNode.prototype );
    Sphere.prototype.constructor = Sphere;

    // TODO: should we allow buffer parameters to change?
    Sphere.prototype.setupBuffers = function () {
        var gl = this.gl;
        this.positionBuffer = gl.createBuffer();
        this.normalBuffer = gl.createBuffer();
        this.indexBuffer = gl.createBuffer();
        this.textureBuffer = gl.createBuffer();

        var positionData = [];
        var normalData = [];
        var indexData = [];
        var textureData = [];

        var latitudeSamples = this.latitudeStrips + 1;
        var longitudeSamples = this.longitudeStrips;

        var thetaIndex;

        for ( var phiIndex = 0; phiIndex < latitudeSamples; phiIndex++ ) {
            var v = phiIndex / (latitudeSamples - 1.0);
            var phi = ( v - 0.5 ) * Math.PI;

            var cosPhi = Math.cos( phi );
            var sinPhi = Math.sin( phi );

            for ( thetaIndex = 0; thetaIndex <= longitudeSamples; thetaIndex++ ) {
                var u = thetaIndex / longitudeSamples;
                var theta = u * 2 * Math.PI;

                var cosTheta = Math.cos( theta );
                var sinTheta = Math.sin( theta );

                var x = sinTheta * cosPhi;
                var y = sinTheta * sinPhi;
                var z = cosTheta;

                positionData.push( x * this.radius );
                positionData.push( y * this.radius );
                positionData.push( z * this.radius );

                normalData.push( x );
                normalData.push( y );
                normalData.push( z );

                textureData.push( u );
                textureData.push( v );
            }

            if ( phiIndex > 0 ) {
                var baseA = (phiIndex - 1) * (longitudeSamples + 1);
                var baseB = phiIndex * (longitudeSamples + 1);

                for ( thetaIndex = 1; thetaIndex <= longitudeSamples; thetaIndex++ ) {
                    // upper-left triangle
                    indexData.push( baseA + thetaIndex - 1 );
                    indexData.push( baseB + thetaIndex - 1 );
                    indexData.push( baseB + thetaIndex );

                    // lower-right triangle
                    indexData.push( baseA + thetaIndex - 1 );
                    indexData.push( baseB + thetaIndex );
                    indexData.push( baseA + thetaIndex );
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

    Sphere.prototype.renderSelf = function ( args ) {
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

    Sphere.prototype.intersect = function ( ray, epsilon ) {
        epsilon = epsilon === undefined ? 0.00001 : epsilon;

        // center is the origin for now, but leaving in computations so that we can change that in the future. optimize away if needed
        var center = new phet.math.Vector3();

        var rayDir = ray.dir;
        var pos = ray.pos;
        var centerToRay = pos.minus( center );

        // basically, we can use the quadratic equation to solve for both possible hit points (both +- roots are the hit points)
        var tmp = rayDir.dot( centerToRay );
        var centerToRayDistSq = centerToRay.magnitudeSquared();
        var det = 4 * tmp * tmp - 4 * ( centerToRayDistSq - this.radius * this.radius );
        if ( det < epsilon ) {
            // ray misses sphere entirely
            return null;
        }

        var base = rayDir.dot( center ) - rayDir.dot( pos );
        var sqt = Math.sqrt( det ) / 2;

        // the "first" entry point distance into the sphere. if we are inside the sphere, it is behind us
        var ta = base - sqt;

        // the "second" entry point distance
        var tb = base + sqt;

        if ( tb < epsilon ) {
            // sphere is behind ray, so don't return an intersection
            return null;
        }

        var hitPositionB = ray.pointAtDistance( tb );
        var normalB = hitPositionB.minus( center ).normalized();

        if ( ta < epsilon ) {
            // we are inside the sphere
            // in => out
            return {
                distance: tb,
                hitPoint: hitPositionB,
                normal: normalB.negated(),
                fromOutside: false
            };
        }
        else {
            // two possible hits
            var hitPositionA = ray.pointAtDistance( ta );
            var normalA = hitPositionA.minus( center ).normalized();

            // close hit, we have out => in
            return {
                distance: ta,
                hitPoint: hitPositionA,
                normal: normalA,
                fromOutside: true
            };
        }
    }
})();
