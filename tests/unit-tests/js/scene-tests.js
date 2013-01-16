

(function(){
    
    var canvasWidth = 640;
    var canvasHeight = 480;
    
    // takes a snapshot of a scene and stores the pixel data, so that we can compare them
    function snapshot( scene ) {
        var canvas = document.createElement( 'canvas' );
        canvas.width = canvasWidth;
        canvas.height = canvasHeight;
        var context = phet.canvas.initCanvas( canvas );
        scene.renderToCanvas( canvas, context );
        var data = context.getImageData( 0, 0, canvasWidth, canvasHeight );
        console.log( data );
        return data;
    }
    
    // compares two pixel snapshots and uses the qunit's assert to verify they are the same
    function snapshotEquals( a, b, threshold, message ) {
        var isEqual = a.width == b.width && a.height == b.height;
        if( isEqual ) {
            for( var i = 0; i < a.data.length; i++ ) {
                if( a.data[i] != b.data[i] && Math.abs( a.data[i] - b.data[i] ) > threshold ) {
                    isEqual = false;
                    break;
                }
            }
        }
        ok( isEqual, message );
        return isEqual;
    }
    
    // compares the "update" render against a full render in-between a series of steps
    function updateVsFullRender( actions ) {
        var mainScene = new phet.scene.Scene( $( '#main' ) );
        var secondaryScene = new phet.scene.Scene( $( '#secondary' ) );
        
        var mainRoot = mainScene.root;
        var secondaryRoot = secondaryScene.root;
        
        mainRoot.layerType = phet.scene.layers.CanvasLayer;
        secondaryRoot.layerType = phet.scene.layers.CanvasLayer;
        
        for( var i = 0; i < actions.length; i++ ) {
            var action = actions[i];
            action( mainScene );
            action( secondaryScene );
            
            var isEqual = snapshotEquals( snapshot( mainScene ), snapshot( secondaryScene ), 0, 'action #' + i );
            if( !isEqual ) {
                break;
            }
        }
    }
    
    /*---------------------------------------------------------------------------*
    * TESTS BELOW
    *----------------------------------------------------------------------------*/     
    
    module( 'Canvas Scene tests' );
    
    test( 'Canvas 2D Context and Features', function() {
        var canvas = document.createElement( 'canvas' );
        var context = phet.canvas.initCanvas( canvas );
        
        ok( context, 'context' );
        
        var neededMethods = [
            'arc',
            'arcTo',
            'beginPath',
            'bezierCurveTo',
            'clearRect',
            'clip',
            'closePath',
            'fill',
            'fillRect',
            'fillStyle',
            'lineTo',
            'moveTo',
            'rect',
            'restore',
            'quadraticCurveTo',
            'save',
            'setTransform',
            'stroke',
            'strokeRect',
            'strokeStyle'
        ];
        _.each( neededMethods, function( method ) {
            ok( context[method] !== undefined, 'context.' + method );
        } );
    } );
    
    test( 'Text width measurement in canvas', function() {
        var canvas = document.createElement( 'canvas' );
        var context = phet.canvas.initCanvas( canvas );
        var metrics = context.measureText('Hello World');
        ok( metrics.width, 'metrics.width' );
    } );
    
    test( 'Checking Layers and external canvas', function() {
        var scene = new phet.scene.Scene( $( '#main' ) );
        var root = scene.root;
        root.layerType = phet.scene.layers.CanvasLayer;
        
        root.addChild( new phet.scene.nodes.Rectangle( {
            x: 0,
            y: 0,
            width: canvasWidth / 2,
            height: canvasHeight / 2,
            fill: '#ff0000'
        } ) );
        
        var middleRect = new phet.scene.nodes.Rectangle( {
            x: canvasWidth / 4,
            y: canvasHeight / 4,
            width: canvasWidth / 2,
            height: canvasHeight / 2,
            fill: '#00ff00'
        } );
        middleRect.layerType = phet.scene.layers.CanvasLayer;
        
        root.addChild( middleRect );
        
        root.addChild( new phet.scene.nodes.Rectangle( {
            x: canvasWidth / 2,
            y: canvasHeight / 2,
            width: canvasWidth / 2,
            height: canvasHeight / 2,
            fill: '#0000ff'
        } ) );
        
        scene.rebuildLayers();
        scene.updateScene();
        
        snapshot( scene );
        
        equal( scene.layers.length, 3, 'simple layer check' );
        
        var canvas = document.createElement( 'canvas' );
        canvas.width = canvasWidth;
        canvas.height = canvasHeight;
        var context = phet.canvas.initCanvas( canvas );
        $( '#display' ).empty();
        $( '#display' ).append( canvas );
        scene.renderToCanvas( canvas, context );
        
        $( '#display' ).empty();
    } );
    
    test( 'Update vs Full Sequence A', function() {
        updateVsFullRender( [
            function( scene ) {
                scene.root.addChild( new phet.scene.nodes.Rectangle( {
                    x: 0,
                    y: 0,
                    width: canvasWidth / 2,
                    height: canvasHeight / 2,
                    fill: '#000000'
                } ) );
                scene.rebuildLayers();
            }, function( scene ) {
                scene.root.children[0].translate( 20, 20 );
            }
        ] );
    } );
    
    /*---------------------------------------------------------------------------*
    * WebGL
    *----------------------------------------------------------------------------*/        
    
    module( 'WebGL tests' );
    
    test( 'Canvas WebGL Context and Features', function() {
        var canvas = document.createElement( 'canvas' );
        var context = phet.webgl.initWebGL( canvas );
        ok( context, 'context' );
    } );
    
    /*---------------------------------------------------------------------------*
    * Canvas V5 (NEW)
    *----------------------------------------------------------------------------*/        
    
    // v5 canvas additions
    module( 'Bleeding Edge Canvas Support' );
    
    test( 'Canvas 2D v5 Features', function() {
        var canvas = document.createElement( 'canvas' );
        var context = phet.canvas.initCanvas( canvas );
        
        var neededMethods = [
            'addHitRegion',
            'ellipse',
            'resetClip',
            'resetTransform'
        ];
        _.each( neededMethods, function( method ) {
            ok( context[method] !== undefined, 'context.' + method );
        } );
    } );
    
    test( 'Path object support', function() {
        var path = new Path();
    } );
       
    test( 'Text width measurement in canvas', function() {
        var canvas = document.createElement( 'canvas' );
        var context = phet.canvas.initCanvas( canvas );
        var metrics = context.measureText('Hello World');
        _.each( [ 'actualBoundingBoxLeft', 'actualBoundingBoxRight', 'actualBoundingBoxAscent', 'actualBoundingBoxDescent' ], function( method ) {
            ok( metrics[method] !== undefined, 'metrics.' + method );
        } );
    } );
    
})();



