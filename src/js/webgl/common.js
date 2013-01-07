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
})();


