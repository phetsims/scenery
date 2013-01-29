

(function(){
    "use strict";
    
    // $( '#display' ).hide();
    
    var canvasWidth = 320;
    var canvasHeight = 240;
    
    var unicodeTestStrings = [
        "This is a test",
        "Newline\nJaggies?",
        "\u222b",
        "\ufdfa",
        "\u00a7",
        "\u00C1",
        "\u00FF",
        "\u03A9",
        "\u0906",
        "\u79C1",
        "\u9054",
        "A\u030a\u0352\u0333\u0325\u0353\u035a\u035e\u035e",
        "0\u0489",
        "\u2588"
    ];
    
    // takes a snapshot of a scene and stores the pixel data, so that we can compare them
    function snapshot( scene ) {
        var canvas = document.createElement( 'canvas' );
        canvas.width = canvasWidth;
        canvas.height = canvasHeight;
        var context = phet.canvas.initCanvas( canvas );
        scene.renderToCanvas( canvas, context );
        var data = context.getImageData( 0, 0, canvasWidth, canvasHeight );
        return data;
    }
    
    function snapshotToCanvas( snapshot ) {
        var canvas = document.createElement( 'canvas' );
        canvas.width = canvasWidth;
        canvas.height = canvasHeight;
        var context = phet.canvas.initCanvas( canvas );
        context.putImageData( snapshot, 0, 0 );
        $( canvas ).css( 'border', '1px solid black' );
        return canvas;
    }
    
    // compares two pixel snapshots and uses the qunit's assert to verify they are the same
    function snapshotEquals( a, b, threshold, message ) {
        var isEqual = a.width == b.width && a.height == b.height;
        var largestDifference = 0;
        if( isEqual ) {
            for( var i = 0; i < a.data.length; i++ ) {
                if( a.data[i] != b.data[i] && Math.abs( a.data[i] - b.data[i] ) > threshold ) {
                    // console.log( message + ": " + Math.abs( a.data[i] - b.data[i] ) );
                    largestDifference = Math.max( largestDifference, Math.abs( a.data[i] - b.data[i] ) );
                    isEqual = false;
                    // break;
                }
            }
        }
        if( largestDifference > 0 ) {
            var display = $( '#display' );
            // header
            var note = document.createElement( 'h2' );
            $( note ).text( message );
            display.append( note );
            var differenceDiv = document.createElement( 'div' );
            $( differenceDiv ).text( 'Largest pixel color-channel difference: ' + largestDifference );
            display.append( differenceDiv );
            
            display.append( snapshotToCanvas( a ) );
            display.append( snapshotToCanvas( b ) );
            
            // for a line-break
            display.append( document.createElement( 'div' ) );
            
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
        
        for( var i = 0; i < actions.length; i++ ) {
            var action = actions[i];
            action( mainScene );
            mainScene.updateScene();
            
            secondaryScene.clearAllLayers();
            action( secondaryScene );
            secondaryScene.rebuildLayers();
            secondaryScene.renderScene();
            
            var isEqual = snapshotEquals( snapshot( mainScene ), snapshot( secondaryScene ), 0, 'action #' + i );
            if( !isEqual ) {
                break;
            }
        }
    }
    
    function sceneEquals( constructionA, constructionB, message, threshold ) {
        if( threshold === undefined ) {
            threshold = 0;
        }
        
        var sceneA = new phet.scene.Scene( $( '#main' ) );
        var sceneB = new phet.scene.Scene( $( '#secondary' ) );
        
        constructionA( sceneA );
        constructionB( sceneB );
        
        sceneA.renderScene();
        sceneB.renderScene();
        
        var isEqual = snapshotEquals( snapshot( sceneA ), snapshot( sceneB ), threshold, message );
        
        // TODO: consider showing if tests fail
        return isEqual;
    }
    
    function strokeEqualsFill( shapeToStroke, shapeToFill, strokeNodeSetup, message ) {
        sceneEquals( function( scene ) {
            var node = new phet.scene.Node();
            node.setShape( shapeToStroke );
            node.setStroke( '#000000' );
            if( strokeNodeSetup ) { strokeNodeSetup( node ); }
            scene.root.addChild( node );
        }, function( scene ) {
            var node = new phet.scene.Node();
            node.setShape( shapeToFill );
            node.setFill( '#000000' );
            // node.setStroke( '#ff0000' ); // for debugging strokes
            scene.root.addChild( node );
            // node.validateBounds();
            // scene.root.addChild( new phet.scene.Node( {
            //     shape: phet.scene.Shape.bounds( node.getSelfBounds() ),
            //     fill: 'rgba(0,0,255,0.5)'
            // } ) );
        }, message, 128 ); // threshold of 128 due to antialiasing differences between fill and stroke... :(
    }
    
    function testTextBounds( getBoundsOfText, fontDrawingStyles, message ) {
        var precision = 1;
        var title = document.createElement( 'h2' );
        $( title ).text( message );
        $( '#display' ).append( title );
        _.each( unicodeTestStrings, function( testString ) {
            var testBounds = getBoundsOfText( testString, fontDrawingStyles );
            var bestBounds = phet.scene.canvasTextBoundsAccurate( testString, fontDrawingStyles );
            
            var widthOk = Math.abs( testBounds.width() - bestBounds.width() ) < precision;
            var heightOk = Math.abs( testBounds.height() - bestBounds.height() ) < precision;
            var xOk = Math.abs( testBounds.x() - bestBounds.x() ) < precision;
            var yOk = Math.abs( testBounds.y() - bestBounds.y() ) < precision;
            
            var allOk = widthOk && heightOk && xOk && yOk;
            
            ok( widthOk, testString + ' width error: ' + Math.abs( testBounds.width() - bestBounds.width() ) );
            ok( heightOk, testString + ' height error: ' + Math.abs( testBounds.height() - bestBounds.height() ) );
            ok( xOk, testString + ' x error: ' + Math.abs( testBounds.x() - bestBounds.x() ) );
            ok( yOk, testString + ' y error: ' + Math.abs( testBounds.y() - bestBounds.y() ) );
            
            // show any failures
            var pad = 5;
            var scaling = 4; // scale it for display accuracy
            var canvas = document.createElement( 'canvas' );
            canvas.width = Math.ceil( bestBounds.width() + pad * 2 ) * scaling;
            canvas.height = Math.ceil( bestBounds.height() + pad * 2 ) * scaling;
            var context = phet.canvas.initCanvas( canvas );
            context.scale( scaling, scaling );
            context.translate( pad - bestBounds.x(), pad - bestBounds.y() ); // center the text in our bounds
            
            // background bounds
            context.fillStyle = allOk ? '#ccffcc' : '#ffcccc'; // red/green depending on whether it passed
            context.fillRect( testBounds.x(), testBounds.y(), testBounds.width(), testBounds.height() );
            
            // text on top
            context.fillStyle = 'rgba(0,0,0,0.7)';
            context.font = fontDrawingStyles.font;
            context.textAlign = fontDrawingStyles.textAlign;
            context.textBaseline = fontDrawingStyles.textBaseline;
            context.direction = fontDrawingStyles.direction;
            context.fillText( testString, 0, 0 );
            
            $( canvas ).css( 'border', '1px solid black' );
            $( '#display' ).append( canvas );
        } );
    }
    
    function equalsApprox( a, b, message ) {
        ok( Math.abs( a - b ) < 0.0000001, ( message ? message + ': ' : '' ) + a + ' =? ' + b );
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
    
    test( 'Sceneless node handling', function() {
        var a = new phet.scene.Node();
        var b = new phet.scene.Node();
        var c = new phet.scene.Node();
        
        a.setShape( phet.scene.Shape.rectangle( 0, 0, 20, 20 ) );
        c.setShape( phet.scene.Shape.rectangle( 10, 10, 30, 30 ) );
        
        a.addChild( b );
        b.addChild( c );
        
        a.validateBounds();
        
        a.removeChild( b );
        c.addChild( a );
        
        b.validateBounds();
        
        ok( !a.isRooted() );
        
        a.invalidatePaint();
    } );
    
    test( 'Checking Layers and external canvas', function() {
        var scene = new phet.scene.Scene( $( '#main' ) );
        var root = scene.root;
        
        root.addChild( new phet.scene.Node( {
            shape: phet.scene.Shape.rectangle( 0, 0, canvasWidth / 2, canvasHeight / 2 ),
            fill: '#ff0000'
        } ) );
        
        var middleRect = new phet.scene.Node( {
            shape: phet.scene.Shape.rectangle( canvasWidth / 4, canvasHeight / 4, canvasWidth / 2, canvasHeight / 2 ),
            fill: '#00ff00'
        } );
        middleRect.setLayerType( phet.scene.CanvasLayer );
        
        root.addChild( middleRect );
        
        root.addChild( new phet.scene.Node( {
            shape: phet.scene.Shape.rectangle( canvasWidth / 2, canvasHeight / 2, canvasWidth / 2, canvasHeight / 2 ),
            fill: '#0000ff'
        } ) );
        
        scene.updateScene();
        
        equal( scene.layers.length, 3, 'simple layer check' );
    } );
    
    test( 'Update vs Full Basic Clearing Check', function() {
        updateVsFullRender( [
            function( scene ) {
                scene.root.addChild( new phet.scene.Node( {
                    shape: phet.scene.Shape.rectangle( 0, 0, canvasWidth / 2, canvasHeight / 2 ),
                    fill: '#000000'
                } ) );
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
            }, function( scene ) {
                scene.root.children[0].setShape( phet.scene.Shape.rectangle( 0, 0, canvasWidth / 2, canvasHeight / 2 ) );
            }
        ] );
    } );
    
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
            }, function( scene ) {
                scene.root.children[0].translate( canvasWidth / 4, canvasHeight / 4 );
            }
        ] );
    } );
    
    test( 'Correct bounds on rectangle', function() {
        var rectBounds = phet.scene.canvasAccurateBounds( function( context ) { context.fillRect( 100, 100, 200, 200 ); } );
        ok( Math.abs( rectBounds.minX - 100 ) < 0.01, rectBounds.minX );
        ok( Math.abs( rectBounds.minY - 100 ) < 0.01, rectBounds.minY );
        ok( Math.abs( rectBounds.maxX - 300 ) < 0.01, rectBounds.maxX );
        ok( Math.abs( rectBounds.maxY - 300 ) < 0.01, rectBounds.maxY );
    } );
    
    test( 'Consistent and precise bounds range on Text', function() {
        var textBounds = phet.scene.canvasAccurateBounds( function( context ) { context.fillText( 'test string', 0, 0 ); } );
        ok( textBounds.isConsistent, textBounds.toString() );
        
        // precision of 0.001 (or lower given different parameters) is possible on non-Chome browsers (Firefox, IE9, Opera)
        ok( textBounds.precision < 0.15, 'precision: ' + textBounds.precision );
    } );
    
    test( 'ES5 Setter / Getter tests', function() {
        var node = new phet.scene.Node();
        var fill = '#abcdef';
        node.fill = fill;
        equal( node.fill, fill );
        equal( node.getFill(), fill );
        
        var otherNode = new phet.scene.Node( { fill: fill, shape: phet.scene.Shape.rectangle( 0, 0, 10, 10 ) } );
        
        equal( otherNode.fill, fill );
    } );
    
    test( 'Layer change stability', function() {
        var scene = new phet.scene.Scene( $( '#main' ) );
        var root = scene.root;
        
        root.addChild( new phet.scene.Node( {
            shape: phet.scene.Shape.rectangle( 0, 0, canvasWidth / 2, canvasHeight / 2 ),
            fill: '#ff0000'
        } ) );
        
        var middleRect = new phet.scene.Node( {
            shape: phet.scene.Shape.rectangle( canvasWidth / 4, canvasHeight / 4, canvasWidth / 2, canvasHeight / 2 ),
            fill: '#00ff00'
        } );
        
        
        root.addChild( middleRect );
        
        root.addChild( new phet.scene.Node( {
            shape: phet.scene.Shape.rectangle( canvasWidth / 2, canvasHeight / 2, canvasWidth / 2, canvasHeight / 2 ),
            fill: '#0000ff'
        } ) );
        
        scene.updateScene();
        
        var snapshotA = snapshot( scene );
        
        middleRect.setLayerType( phet.scene.CanvasLayer );
        scene.updateScene();
        
        var snapshotB = snapshot( scene );
        
        snapshotEquals( snapshotA, snapshotB, 0, 'Layer change stability' );
    } );
    
    test( 'Piccolo-like behavior', function() {
        var node = new phet.scene.Node();
        
        node.scaleBy( 2 );
        node.translate( 1, 3 );
        node.rotate( Math.PI / 2 );
        node.translate( -31, 21 );
        
        equalsApprox( node.getMatrix().m00(), 0 );
        equalsApprox( node.getMatrix().m01(), -2 );
        equalsApprox( node.getMatrix().m02(), -40 );
        equalsApprox( node.getMatrix().m10(), 2 );
        equalsApprox( node.getMatrix().m11(), 0 );
        equalsApprox( node.getMatrix().m12(), -56 );
        
        equalsApprox( node.x, -40 );
        equalsApprox( node.y, -56 );
        equalsApprox( node.rotation, Math.PI / 2 );
        
        node.translation = new phet.math.Vector2( -5, 7 );
        
        console.log( node.getMatrix().toString() );
        
        equalsApprox( node.getMatrix().m02(), -5 );
        equalsApprox( node.getMatrix().m12(), 7 );
    } );
    
    /*---------------------------------------------------------------------------*
    * Shapes
    *----------------------------------------------------------------------------*/        
    
    (function(){
        module( 'Shapes' );
        
        var Shape = phet.scene.Shape;
        
        function p( x, y ) { return new phet.math.Vector2( x, y ); }
    
        test( 'Verifying Line/Rect', function() {
            var lineWidth = 50;
            // /shapeToStroke, shapeToFill, strokeNodeSetup, message, debugFlag
            var strokeShape = Shape.lineSegment( p( 100, 100 ), p( 300, 100 ) );
            var fillShape = Shape.rectangle( 100, 100 - lineWidth / 2, 200, lineWidth );
            
            strokeEqualsFill( strokeShape, fillShape, function( node ) { node.setLineWidth( lineWidth ); }, QUnit.config.current.testName );
        } );
        
        test( 'Line Segment - butt', function() {
            var styles = new Shape.LineStyles();
            styles.lineWidth = 50;
            
            var strokeShape = Shape.lineSegment( p( 100, 100 ), p( 300, 100 ) );
            var fillShape = strokeShape.getStrokedShape( styles );
            
            strokeEqualsFill( strokeShape, fillShape, function( node ) { node.setLineStyles( styles ); }, QUnit.config.current.testName );
        } );
        
        test( 'Line Segment - square', function() {
            var styles = new Shape.LineStyles();
            styles.lineWidth = 50;
            styles.lineCap = 'square';
            
            var strokeShape = Shape.lineSegment( p( 100, 100 ), p( 300, 100 ) );
            var fillShape = strokeShape.getStrokedShape( styles );
            
            strokeEqualsFill( strokeShape, fillShape, function( node ) { node.setLineStyles( styles ); }, QUnit.config.current.testName );
        } );
        
        test( 'Line Join - Miter', function() {
            var styles = new Shape.LineStyles();
            styles.lineWidth = 30;
            styles.lineJoin = 'miter';
            
            var strokeShape = new Shape();
            strokeShape.moveTo( 70, 70 );
            strokeShape.lineTo( 140, 200 );
            strokeShape.lineTo( 210, 70 );
            var fillShape = strokeShape.getStrokedShape( styles );
            
            strokeEqualsFill( strokeShape, fillShape, function( node ) { node.setLineStyles( styles ); }, QUnit.config.current.testName );
        } );
        
        test( 'Line Join - Miter - Closed', function() {
            var styles = new Shape.LineStyles();
            styles.lineWidth = 30;
            styles.lineJoin = 'miter';
            
            var strokeShape = new Shape();
            strokeShape.moveTo( 70, 70 );
            strokeShape.lineTo( 140, 200 );
            strokeShape.lineTo( 210, 70 );
            strokeShape.close();
            var fillShape = strokeShape.getStrokedShape( styles );
            
            strokeEqualsFill( strokeShape, fillShape, function( node ) { node.setLineStyles( styles ); }, QUnit.config.current.testName );
        } );
        
        test( 'Line Join - Bevel - Closed', function() {
            var styles = new Shape.LineStyles();
            styles.lineWidth = 30;
            styles.lineJoin = 'bevel';
            
            var strokeShape = new Shape();
            strokeShape.moveTo( 70, 70 );
            strokeShape.lineTo( 140, 200 );
            strokeShape.lineTo( 210, 70 );
            strokeShape.close();
            var fillShape = strokeShape.getStrokedShape( styles );
            
            strokeEqualsFill( strokeShape, fillShape, function( node ) { node.setLineStyles( styles ); }, QUnit.config.current.testName );
        } );
        
        test( 'Rect', function() {
            var styles = new Shape.LineStyles();
            styles.lineWidth = 30;
            
            var strokeShape = Shape.rectangle( 40, 40, 150, 150 );
            var fillShape = strokeShape.getStrokedShape( styles );
            
            strokeEqualsFill( strokeShape, fillShape, function( node ) { node.setLineStyles( styles ); }, QUnit.config.current.testName );
        } );
        
        test( 'Manual Rect', function() {
            var styles = new Shape.LineStyles();
            styles.lineWidth = 30;
            
            var strokeShape = new Shape();
            strokeShape.moveTo( 40, 40 );
            strokeShape.lineTo( 190, 40 );
            strokeShape.lineTo( 190, 190 );
            strokeShape.lineTo( 40, 190 );
            strokeShape.lineTo( 40, 40 );
            strokeShape.close();
            var fillShape = strokeShape.getStrokedShape( styles );
            
            strokeEqualsFill( strokeShape, fillShape, function( node ) { node.setLineStyles( styles ); }, QUnit.config.current.testName );
        } );
        
        test( 'Hex', function() {
            var styles = new Shape.LineStyles();
            styles.lineWidth = 30;
            
            var strokeShape = Shape.regularPolygon( 6, 100 ).transformed( phet.math.Matrix3.translation( 130, 130 ) );
            var fillShape = strokeShape.getStrokedShape( styles );
            
            strokeEqualsFill( strokeShape, fillShape, function( node ) { node.setLineStyles( styles ); }, QUnit.config.current.testName );
        } );
        
        test( 'Overlap', function() {
            var styles = new Shape.LineStyles();
            styles.lineWidth = 30;
            
            var strokeShape = new Shape();
            strokeShape.moveTo( 40, 40 );
            strokeShape.lineTo( 200, 200 );
            strokeShape.lineTo( 40, 200 );
            strokeShape.lineTo( 200, 40 );
            strokeShape.lineTo( 100, 0 );
            strokeShape.close();
            var fillShape = strokeShape.getStrokedShape( styles );
            
            strokeEqualsFill( strokeShape, fillShape, function( node ) { node.setLineStyles( styles ); }, QUnit.config.current.testName );
        } );
        
        var miterMagnitude = 160;
        var miterAnglesInDegrees = [5, 8, 10, 11.5, 13, 20, 24, 30, 45];
        
        _.each( miterAnglesInDegrees, function( miterAngle ) {
            var miterAngleRadians = miterAngle * Math.PI / 180;
            test( 'Miter limit angle (degrees): ' + miterAngle + ' would change at ' + 1 / Math.sin( miterAngleRadians / 2 ), function() {
                var styles = new Shape.LineStyles();
                styles.lineWidth = 30;
                
                var strokeShape = new Shape();
                var point = new phet.math.Vector2( 40, 100 );
                strokeShape.moveTo( point );
                point = point.plus( phet.math.Vector2.X_UNIT.times( miterMagnitude ) );
                strokeShape.lineTo( point );
                point = point.plus( phet.math.Vector2.createPolar( miterMagnitude, miterAngleRadians ).negated() );
                strokeShape.lineTo( point );
                var fillShape = strokeShape.getStrokedShape( styles );
                
                strokeEqualsFill( strokeShape, fillShape, function( node ) { node.setLineStyles( styles ); }, QUnit.config.current.testName );
            } );
        } );
        
        test( 'Overlapping rectangles', function() {
            var styles = new Shape.LineStyles();
            styles.lineWidth = 30;
            
            var strokeShape = new Shape();
            strokeShape.rect( 40, 40, 100, 100 );
            strokeShape.rect( 50, 50, 100, 100 );
            strokeShape.rect( 80, 80, 100, 100 );
            var fillShape = strokeShape.getStrokedShape( styles );
            
            strokeEqualsFill( strokeShape, fillShape, function( node ) { node.setLineStyles( styles ); }, QUnit.config.current.testName );
        } );
        
        test( 'Line segment winding', function() {
            var line = new Shape.Segment.Line( p( 0, 0 ), p( 2, 2 ) );
            
            equal( line.windingIntersection( new phet.math.Ray2( p( 0, 1 ), p( 1, 0 ) ) ), 1 );
            equal( line.windingIntersection( new phet.math.Ray2( p( 0, 5 ), p( 1, 0 ) ) ), 0 );
            equal( line.windingIntersection( new phet.math.Ray2( p( 1, 0 ), p( 0, 1 ) ) ), -1 );
            equal( line.windingIntersection( new phet.math.Ray2( p( 0, 0 ), p( 1, 1 ).normalized() ) ), 0 );
            equal( line.windingIntersection( new phet.math.Ray2( p( 0, 1 ), p( 1, 1 ).normalized() ) ), 0 );
        } );
        
        test( 'Rectangle hit testing', function() {
            var shape = Shape.rectangle( 0, 0, 1, 1 );
            
            equal( shape.containsPoint( p( 0.2, 0.3 ) ), true, '0.2, 0.3' );
            equal( shape.containsPoint( p( 0.5, 0.5 ) ), true, '0.5, 0.5' );
            equal( shape.containsPoint( p( 1.5, 0.5 ) ), false, '1.5, 0.5' );
            equal( shape.containsPoint( p( -0.5, 0.5 ) ), false, '-0.5, 0.5' );
        } );
        
        test( 'Bezier Offset', function() {
            var styles = new Shape.LineStyles();
            styles.lineWidth = 30;
            
            var strokeShape = new Shape();
            strokeShape.moveTo( 40, 40 );
            strokeShape.quadraticCurveTo( 100, 200, 160, 40 );
            // strokeShape.close();
            var fillShape = strokeShape.getStrokedShape( styles );
            
            strokeEqualsFill( strokeShape, fillShape, function( node ) { node.setLineStyles( styles ); }, QUnit.config.current.testName );
        } );
        
        test( 'Bezier Edge Case (literally)', function() {
            var styles = new Shape.LineStyles();
            styles.lineWidth = 30;
            
            var strokeShape = new Shape();
            strokeShape.moveTo( 40, 40 );
            strokeShape.quadraticCurveTo( 200, 200, 200, 180 );
            // strokeShape.close();
            var fillShape = strokeShape.getStrokedShape( styles );
            
            strokeEqualsFill( strokeShape, fillShape, function( node ) { node.setLineStyles( styles ); }, QUnit.config.current.testName );
        } );
    })();
    
    /*---------------------------------------------------------------------------*
    * Text
    *----------------------------------------------------------------------------*/        
    
    module( 'Text' );
    
    test( 'Canvas Accurate Text Bounds', function() {
        testTextBounds( phet.scene.canvasTextBoundsAccurate, {
            font: '10px sans-serif',
            textAlign: 'left', // left is not the default, 'start' is
            textBaseline: 'alphabetic',
            direction: 'ltr'
        }, QUnit.config.current.testName );
    } );
    
    /*---------------------------------------------------------------------------*
    * DOM
    *----------------------------------------------------------------------------*/        
    
    module( 'DOM Layers' );
    
    test( 'DOM Test', function() {
        updateVsFullRender( [
            function( scene ) {
                var node = new phet.scene.Node();
                node.setShape( phet.scene.Shape.rectangle( 0, 0, canvasWidth / 3, canvasHeight / 3 ) );
                node.setFill( '#ff0000' );
                node.setStroke( '#000000' );
                scene.root.addChild( node );
                
                var domNode = new phet.scene.Node();
                domNode.setLayerType( phet.scene.DOMLayer );
                node.addChild( domNode );
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



