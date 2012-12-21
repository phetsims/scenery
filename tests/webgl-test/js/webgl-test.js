$( document ).ready( function () {
    var model = phet.moleculeshapes.model;

    var Property = phet.model.Property;
    var Dimension2 = phet.math.Dimension2;
    var Vector3 = phet.math.Vector3;

    // whether we should attempt to use WebGL first
    var preferGL = true;

    var alpha = '';
    var beta = '';
    var gamma = '';

    // whether we are actually using WebGL
    var useGL;

    // frustum (camera) properties
    var fieldOfViewDegrees = 45 / 2;
    var nearPlane = 1;
    var farPlane = 1000;
    var mainTranslation = new Vector3( 0, 0, -40 );

    var mainCanvas = $( '#mainCanvas' )[0];
    var uiCanvas = $( '#uiCanvas' )[0];

    var molecule = new model.VSEPRMolecule(); // no extra args needed, no bond override

    // stop text selection on the canvas
    var emptyFunction = function () { return false; };
    mainCanvas.onselectstart = emptyFunction;
    uiCanvas.onselectstart = emptyFunction;

    phet.assert( mainCanvas.getContext );

    var shaderProgram;
    var gl, uiContext, mainContext;

    function initShaders() {
        var fragmentShader = phet.webgl.getShaderFromDOM( gl, "shader-fs" );
        var vertexShader = phet.webgl.getShaderFromDOM( gl, "shader-vs" );

        shaderProgram = gl.createProgram();
        gl.attachShader( shaderProgram, vertexShader );
        gl.attachShader( shaderProgram, fragmentShader );
        gl.linkProgram( shaderProgram );

        if ( !gl.getProgramParameter( shaderProgram, gl.LINK_STATUS ) ) {
            alert( "Could not initialise shaders" );
        }

        gl.useProgram( shaderProgram );

        shaderProgram.vertexPositionAttribute = gl.getAttribLocation( shaderProgram, "aVertexPosition" );
        gl.enableVertexAttribArray( shaderProgram.vertexPositionAttribute );

        shaderProgram.pMatrixUniform = gl.getUniformLocation( shaderProgram, "uPMatrix" );
        shaderProgram.mvMatrixUniform = gl.getUniformLocation( shaderProgram, "uMVMatrix" );
        shaderProgram.inverseTransposeMatrixUniform = gl.getUniformLocation( shaderProgram, "uInverseTransposeMatrix" );
        shaderProgram.atomColor = gl.getUniformLocation( shaderProgram, "atomColor" );

        shaderProgram.normalAttribute = gl.getAttribLocation( shaderProgram, "aNormal" );
        gl.enableVertexAttribArray( shaderProgram.normalAttribute );
    }

    if ( preferGL ) {
        try {
            gl = phet.webgl.initWebGL( mainCanvas );
            useGL = true;
        }
        catch( e ) {
            // WebGL context creation failed
            useGL = false;
        }
    }
    if ( !preferGL || !useGL ) {
        mainContext = phet.canvas.initCanvas( uiCanvas );
    }
    uiContext = phet.canvas.initCanvas( uiCanvas );

    if ( useGL ) {
        initShaders();

        gl.clearColor( 0.0, 0.0, 0.0, 1.0 );

        gl.enable( gl.DEPTH_TEST );
        gl.enable( gl.BLEND );
        gl.blendFunc( gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA );
    }

    // property, updated whenever the canvas is resized
    var canvasSize = new Property( new Dimension2( mainCanvas.width, mainCanvas.height ) );

    // plain dimension
    var stageSize = new Dimension2( 1008, 676 ); // dimensions from the desktop Molecule Shapes version

    var canvasTransform = new phet.math.StageCenteringCanvasTransform( canvasSize, stageSize );

    var scene = new phet.webgl.GLNode();
    scene.transform.append( phet.math.Matrix4.translation( mainTranslation.x, mainTranslation.y, mainTranslation.z ) );

    var moleculeNode;
    if ( useGL ) {
        moleculeNode = new phet.moleculeshapes.view.GLMoleculeNode( gl, molecule );
    }
    else {
        moleculeNode = new phet.moleculeshapes.view.CanvasMoleculeNode( mainContext, molecule );
    }
    scene.addChild( moleculeNode );

    /*---------------------------------------------------------------------------*
     * molecule setup
     *----------------------------------------------------------------------------*/

    // start with two single bonds
    var centralAtom = new model.PairGroup( new Vector3(), false, false );
    molecule.addCentralAtom( centralAtom );
    molecule.addGroupAndBond( new model.PairGroup( new Vector3( 8, 0, 3 ).normalized().times( model.PairGroup.BONDED_PAIR_DISTANCE ), false, false ), centralAtom, 1 );
    var vector3D = new Vector3( 2, 8, -5 );
    molecule.addGroupAndBond( new model.PairGroup( vector3D.normalized().times( model.PairGroup.BONDED_PAIR_DISTANCE ), false, false ), centralAtom, 1 );

    var rot = 0;
    var lastTime = 0;
    var timeElapsed = 0;

    function getSceneProjectionMatrix() {
        var fieldOfViewRadians = ( fieldOfViewDegrees / 180 * Math.PI );
        var correctedFieldOfViewRadians = Math.atan( canvasTransform.fieldOfViewYFactor() * Math.tan( fieldOfViewRadians ) );

        return phet.math.Matrix4.gluPerspective( correctedFieldOfViewRadians,
                                                 canvasSize.get().width / canvasSize.get().height,
                                                 nearPlane, farPlane );
    }

    function debuggingText() {
        return canvasSize.get().toString() + " window.orientation = " + window.orientation +
               "<br>alpha: " + alpha +
               "<br>beta: " + beta +
               "<br>gamma: " + gamma +
               "<br>backingScale: " + phet.canvas.backingScale( uiContext ) +
               "<br>window.innerWidth: " + window.innerWidth +
               "<br>window.innerHeight: " + window.innerHeight;
    }

    function drawMainCanvas() {
        // Only continue if WebGL is available and working
        if ( useGL ) {
            gl.viewportWidth = mainCanvas.width;
            gl.viewportHeight = mainCanvas.height;

            gl.viewport( 0, 0, gl.viewportWidth, gl.viewportHeight );
            gl.clear( gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT );

            var projectionMatrix = getSceneProjectionMatrix();
            gl.uniformMatrix4fv( shaderProgram.pMatrixUniform, false, projectionMatrix.entries );

            var args = new phet.webgl.GLNode.RenderState();
            args.transformAttribute = shaderProgram.mvMatrixUniform;
            args.inverseTransposeAttribute = shaderProgram.inverseTransposeMatrixUniform;
            args.positionAttribute = shaderProgram.vertexPositionAttribute;
            args.normalAttribute = shaderProgram.normalAttribute;
            args.gl = gl;
            args.useGL = useGL;
            args.shaderProgram = shaderProgram;

//            scene.transform.append( phet.math.Matrix4.rotationY( (Math.PI / 2 * timeElapsed) ) );

            scene.render( args );
        }
        else {
            var mappingMatrix = getSceneProjectionMatrix();
            mainContext.save();
            canvasTransform.transform.get().applyToCanvasContext( uiContext );
            mainContext.clearRect( 0, 0, stageSize.width, stageSize.height );

            mainContext.beginPath();
            mainContext.rect( 0, 0, stageSize.width, stageSize.height );
            mainContext.clip();

            mainContext.strokeStyle = '#FFFFFF';

            phet.util.foreach( molecule.getAtoms(), function ( atom ) {
                var position = mappingMatrix.timesVector4( atom.position.get().plus( mainTranslation ).toVector4() );
                mainContext.beginPath();
                mainContext.arc( stageSize.width * (position.x / position.w + 1) / 2,
                                 stageSize.height * (1 - (position.y / position.w + 1) / 2),
                                 50, 0, Math.PI * 2 );
                mainContext.stroke();
            } );

            mainContext.restore();
        }

        $( '#topleft' ).html( 'FPS: ' + Math.round( 1 / timeElapsed ) + "<br>" + debuggingText() );
    }

    function drawUICanvas() {
        uiContext.font = '20px sans-serif';
        uiContext.fillStyle = '#FFFFFF';

        uiContext.save();
        canvasTransform.transform.get().applyToCanvasContext( uiContext );
        uiContext.fillText( 'Canvas text in bottom-left', 10, stageSize.height - 10 );
        uiContext.restore();


        uiContext.strokeStyle = '#00FF00';
        uiContext.lineWidth = 1;

        var topLeft = canvasTransform.transform.get().transformPosition3( new phet.math.Vector3( 0, 0, 0 ) );
        var bottomRight = canvasTransform.transform.get().transformPosition3( new phet.math.Vector3( stageSize.width, stageSize.height, 0 ) );
        uiContext.strokeRect( topLeft.x, topLeft.y, bottomRight.x - topLeft.x, bottomRight.y - topLeft.y );
    }

    function animate() {
        molecule.update( timeElapsed );
    }

    function tick() {
        window.requestAnimationFrame( tick, mainCanvas );
        var timeNow = new Date().getTime();
        if ( lastTime != 0 ) {
            timeElapsed = (timeNow - lastTime) / 1000.0;
        }
        lastTime = timeNow;

        animate();
        drawMainCanvas();
    }

    var usePointerCursor = false;
    var cursorX = 0;
    var cursorY = 0;

    var updateCursor = function ( x, y ) {
        cursorX = x;
        cursorY = y;

        if ( usePointerCursor ) {
            $( uiCanvas ).css( "cursor", "pointer" );
        }
        else {
            $( uiCanvas ).css( "cursor", "auto" );
        }
    };

    var resizer = function () {
        mainCanvas.width = uiCanvas.width = window.innerWidth;
        mainCanvas.height = uiCanvas.height = window.innerHeight;
        if ( useGL ) {
            gl.viewportWidth = mainCanvas.width;
            gl.viewportHeight = mainCanvas.height;
        }
        canvasSize.set( new Dimension2( mainCanvas.width, mainCanvas.height ) );
        drawMainCanvas();
        drawUICanvas();
    };
    $( window ).resize( resizer );

    window.addEventListener( "deviceorientation", function ( event ) {
        alpha = event.alpha;
        beta = event.beta;
        gamma = event.gamma;
    }, true );

    tick();

    var moveListener = function ( x, y ) {
        updateCursor( x, y );
    };

    var downListener = function ( x, y ) {
        updateCursor( x, y );

        var v = new Vector3( Math.random() - 0.5, Math.random() - 0.5, Math.random() - 0.5 );
        molecule.addGroupAndBond( new model.PairGroup( v.normalized().times( model.PairGroup.BONDED_PAIR_DISTANCE ), false, false ), centralAtom, 1 );
    };

    var upListener = function ( x, y ) {
        updateCursor( x, y );
    };

    var touchFromJQueryEvent = function ( evt ) {
        return evt.originalEvent.targetTouches[0];
    };

    $( uiCanvas ).bind( "mousemove", function ( evt ) {
        evt.preventDefault();
        moveListener( evt.pageX, evt.pageY );
    } );
    $( uiCanvas ).bind( "mousedown", function ( evt ) {
        evt.preventDefault();
        downListener( evt.pageX, evt.pageY );
    } );
    $( uiCanvas ).bind( "mouseup", function ( evt ) {
        evt.preventDefault();
        upListener( evt.pageX, evt.pageY );
    } );
    $( uiCanvas ).bind( "touchmove", function ( evt ) {
        evt.preventDefault();
        var touch = touchFromJQueryEvent( evt );
        moveListener( touch.pageX, touch.pageY );
    } );
    $( uiCanvas ).bind( "touchstart", function ( evt ) {
        evt.preventDefault();
        var touch = touchFromJQueryEvent( evt );
        downListener( touch.pageX, touch.pageY );
    } );
    $( uiCanvas ).bind( "touchend", function ( evt ) {
        evt.preventDefault();
        var touch = touchFromJQueryEvent( evt );
        upListener( touch.pageX, touch.pageY );
    } );
    $( uiCanvas ).bind( "touchcancel", function ( evt ) {
        evt.preventDefault();
        var touch = touchFromJQueryEvent( evt );
        upListener( touch.pageX, touch.pageY );
    } );
    resizer();
} );