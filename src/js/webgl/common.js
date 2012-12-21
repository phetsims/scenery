// Copyright 2002-2012, University of Colorado

var phet = phet || {};
phet.webgl = phet.webgl || {};

(function () {
    phet.webgl.getShaderFromDOM = function ( gl, id ) {
        var shaderScript = document.getElementById( id );
        if ( !shaderScript ) {
            throw new Error( "shader DOM not found: for id=" + id );
        }

        var str = "";
        var k = shaderScript.firstChild;
        while ( k ) {
            if ( k.nodeType == 3 ) {
                str += k.textContent;
            }
            k = k.nextSibling;
        }

        var shader;
        if ( shaderScript.type == "x-shader/x-fragment" ) {
            shader = gl.createShader( gl.FRAGMENT_SHADER );
        }
        else if ( shaderScript.type == "x-shader/x-vertex" ) {
            shader = gl.createShader( gl.VERTEX_SHADER );
        }
        else {
            throw new Error( "shader DOM type not recognized: " + shaderScript.type );
        }

        gl.shaderSource( shader, str );
        gl.compileShader( shader );

        if ( !gl.getShaderParameter( shader, gl.COMPILE_STATUS ) ) {
            throw new Error( gl.getShaderInfoLog( shader ) );
        }

        return shader;
    };

    phet.webgl.initWebGL = function ( canvas ) {
        // Initialize the global variable gl to null.
        var gl = null;

        try {
            // Try to grab the standard context. If it fails, fallback to experimental.
            gl = canvas.getContext( "webgl" ) || canvas.getContext( "experimental-webgl" );
        }
        catch( e ) {}

        // If we don't have a GL context, give up now
        if ( !gl ) {
            // TODO: show a visual display
            throw new Error( "Unable to initialize WebGL. Your browser may not support it." );
        }

        return gl;
    };

    /*---------------------------------------------------------------------------*
     * window.requestAnimationFrame polyfill, by Erik Moller (http://my.opera.com/emoller/blog/2011/12/20/requestanimationframe-for-smart-er-animating)
     * referenced by initial Paul Irish article at http://paulirish.com/2011/requestanimationframe-for-smart-animating/
     *----------------------------------------------------------------------------*/
    ( function () {
        var lastTime = 0;
        var vendors = ['ms', 'moz', 'webkit', 'o'];
        for ( var x = 0; x < vendors.length && !window.requestAnimationFrame; ++x ) {
            window.requestAnimationFrame = window[vendors[x] + 'RequestAnimationFrame'];
            window.cancelAnimationFrame =
            window[vendors[x] + 'CancelAnimationFrame'] || window[vendors[x] + 'CancelRequestAnimationFrame'];
        }

        if ( !window.requestAnimationFrame ) {
            window.requestAnimationFrame = function ( callback, element ) {
                console.log( 'window.requestAnimationFrame' );
                var currTime = new Date().getTime();
                var timeToCall = Math.max( 0, 16 - (currTime - lastTime) );
                var id = window.setTimeout( function () { callback( currTime + timeToCall ); },
                                            timeToCall );
                lastTime = currTime + timeToCall;
                return id;
            };
        }

        if ( !window.cancelAnimationFrame ) {
            window.cancelAnimationFrame = function ( id ) {
                clearTimeout( id );
            };
        }
    }() );
})();


