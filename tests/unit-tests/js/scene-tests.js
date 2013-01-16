

(function(){
    
    var canvasWidth = 320;
    var canvasHeight = 240;
    
    // takes a snapshot of a scene and stores the pixel data, so that we can compare them
    function snapshot( scene, debugFlag ) {
        var canvas = document.createElement( 'canvas' );
        canvas.width = canvasWidth;
        canvas.height = canvasHeight;
        var context = phet.canvas.initCanvas( canvas );
        scene.renderToCanvas( canvas, context );
        var data = context.getImageData( 0, 0, canvasWidth, canvasHeight );
        if( debugFlag ) {
            $( '#display' ).append( canvas );
            $( canvas ).css( 'border', '1px solid black' );
        }
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
    function updateVsFullRender( actions, debugFlag ) {
        var mainScene = new phet.scene.Scene( $( '#main' ) );
        var secondaryScene = new phet.scene.Scene( $( '#secondary' ) );
        
        var mainRoot = mainScene.root;
        var secondaryRoot = secondaryScene.root;
        
        for( var i = 0; i < actions.length; i++ ) {
            var action = actions[i];
            action( mainScene );
            mainScene.updateScene();
            
            secondaryScene.clearAllLayers();
            action( secondaryScene );
            secondaryScene.rebuildLayers();
            secondaryScene.renderScene();
            
            if( debugFlag ) {
                var note = document.createElement( 'div' );
                $( note ).text( 'Action ' + i );
                $( '#display' ).append( note );
            }
            
            var isEqual = snapshotEquals( snapshot( mainScene, debugFlag ), snapshot( secondaryScene, debugFlag ), 0, 'action #' + i );
            if( !isEqual ) {
                break;
            }
        }
    }
    
    /*---------------------------------------------------------------------------*
    * TESTS BELOW
    *----------------------------------------------------------------------------*/     
    
    module( 'Canvas Scene Regression' );
    
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
            'isPointInPath',
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
        middleRect.setLayerType( phet.scene.layers.CanvasLayer );
        
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
        
        equal( scene.layers.length, 3, 'simple layer check' );
    } );
    
    test( 'Update vs Full Basic Clearing Check', function() {
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
    
    test( 'Update vs Full Self-Bounds increase', function() {
        updateVsFullRender( [
            function( scene ) {
                var node = new phet.scene.Node();
                node.setShape( phet.scene.Shape.rectangle( 0, 0, canvasWidth / 3, canvasHeight / 3 ) );
                node.setFill( '#ff0000' );
                node.setStroke( '#000000' );
                scene.root.addChild( node );
                
                scene.rebuildLayers();
            }, function( scene ) {
                scene.root.children[0].setShape( phet.scene.Shape.rectangle( 0, 0, canvasWidth / 2, canvasHeight / 2 ) );
            }
        ] );
    } );
    
    module( 'Canvas Scene TODO' );
    
    test( 'Update vs Full Stroke Repaint', function() {
        updateVsFullRender( [
            function( scene ) {
                // TODO: clearer way of specifying parameters
                var node = new phet.scene.Node();
                node.setShape( phet.scene.Shape.rectangle( 15, 15, canvasWidth / 2, canvasHeight / 2 ) );
                node.setFill( '#ff0000' );
                node.setStroke( '#000000' );
                node.setLineWidth( 10 );
                scene.root.addChild( node );
                
                scene.rebuildLayers();
            }, function( scene ) {
                scene.root.children[0].translate( canvasWidth / 4, canvasHeight / 4 );
            }
        ] );
    } );
    
    /*---------------------------------------------------------------------------*
    * Miscellaneous HTML / JS
    *----------------------------------------------------------------------------*/        
    
    module( 'Miscellaneous' );
    
    test( 'ES5 Object.defineProperty get/set', function() {
        var ob = { _key: 5 };
        Object.defineProperty( ob, 'key', {
            enumerable: true,
            configurable: true,
            get: function() { return this._key; },
            set: function( val ) { this._key = val; }
        } );
        ob.key += 1;
        equal( ob._key, 6, 'incremented object value' );
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



