
var phet = phet || {};
phet.tests = phet.tests || {};

(function(){
    
    var backgroundSize = 300;
    var count = 500;
    
    phet.tests.sceneLayeringTests = function( main, useLayers ) {
        var scene = new phet.scene.Scene( main );
        var root = scene.root;
        
        function randomizeTranslation( node ) {
            node.setTranslation( ( Math.random() - 0.5 ) * backgroundSize, ( Math.random() - 0.5 ) * backgroundSize );
        }
        
        function buildShapes( color ) {
            var background = new phet.scene.Node();
            if( useLayers && color == 'rgba(0,255,0,0.7)' ) {
                background.setLayerType( phet.scene.layers.CanvasLayer );
            }
            for( var i = 0; i < count; i++ ) {
                var node = new phet.scene.Node();
                var radius = 10;
                
                // regular polygon
                node.setShape( phet.scene.Shape.regularPolygon( 6, radius ) );
                
                node.setFill( color );
                node.setStroke( '#000000' );
                
                randomizeTranslation( node );
                
                background.addChild( node );
            }
            return background;
        }
        
        var reds = buildShapes( 'rgba(255,0,0,0.7)' );
        var greens = buildShapes( 'rgba(0,255,0,0.7)' );
        var blues = buildShapes( 'rgba(0,0,255,0.7)' );
        
        root.addChild( reds );
        root.addChild( greens );
        root.addChild( blues );
        
        // center the root
        root.translate( main.width() / 2, main.height() / 2 );
        
        window.root = root;
        window.scene = scene;
        
        // return step function
        return function( timeElapsed ) {
            greens.rotate( timeElapsed );
            scene.updateScene();
        }
    };
    
    phet.tests.easelLayeringTests = function( main ) {
        var canvas = document.createElement( 'canvas' );
        canvas.id = 'easel-canvas';
        canvas.width = main.width();
        canvas.height = main.height();
        main.append( canvas );

        var stage = new createjs.Stage( canvas );
        
        function buildShapes( color ) {
            var background = new createjs.Container();
            stage.addChild( background );
            
            for( var i = 0; i < count; i++ ) {
                var shape = new createjs.Shape();
                var radius = 10;
                
                shape.graphics.beginFill( color ).beginStroke( '#000000' ).drawPolyStar( 0, 0, radius, 6, 0, 0 );
                
                shape.x = ( Math.random() - 0.5 ) * backgroundSize;
                shape.y = ( Math.random() - 0.5 ) * backgroundSize;
                
                background.addChild( shape );
            }
            
            background.x = main.width() / 2;
            background.y = main.height() / 2;
            
            return background;
        }
        
        var reds = buildShapes( 'rgba(255,0,0,0.7)' );
        var greens = buildShapes( 'rgba(0,255,0,0.7)' );
        var blues = buildShapes( 'rgba(0,0,255,0.7)' );
        
        stage.addChild( reds );
        stage.addChild( greens );
        stage.addChild( blues );
        
        // return step function
        return function( timeElapsed ) {
            greens.rotation += timeElapsed * 180 / Math.PI;
            stage.update();
        }
    };  
    
    phet.tests.customLayeringTests = function( main ) {
        var zIndex = 0;
        var radius = 10;
        
        var points = _.map( _.range( 6 ), function( n ) {
            var theta = 2 * Math.PI * n / 6;
            return new phet.math.Vector2( radius * Math.cos( theta ), radius * Math.sin( theta ) );
        } );
        
        var redSeed = _.map( _.range( count * 2 ), function( boo ) { return ( Math.random() - 0.5 ) * backgroundSize; } );
        var greenSeed = _.map( _.range( count * 2 ), function( boo ) { return ( Math.random() - 0.5 ) * backgroundSize; } );
        var blueSeed = _.map( _.range( count * 2 ), function( boo ) { return ( Math.random() - 0.5 ) * backgroundSize; } );
        
        function addCanvas() {
            var canvas = document.createElement( 'canvas' );
            canvas.width = main.width();
            canvas.height = main.height();
            $( canvas ).css( 'position', 'absolute' );
            $( canvas ).css( 'x-index', zIndex );
            zIndex += 1;
            main.append( canvas );
            
            return phet.canvas.initCanvas( canvas );
        }
        
        var redContext = addCanvas();
        var greenContext = addCanvas();
        var blueContext = addCanvas();
        
        function drawShapes( context, color, seed, rotation ) {
            // center the transform
            context.setTransform( 1, 0, 0, 1, main.width() / 2, main.height() / 2 );
            
            if( rotation != 0 ) {
                context.rotate( rotation );
            }
            
            context.fillStyle = color;
            context.strokeStyle = '#000000';
            
            for( var i = 0; i < count; i++ ) {
                var xOffset = seed[i*2];
                var yOffset = seed[i*2+1];
                
                context.beginPath();
                context.moveTo( points[0].x + xOffset, points[0].y + yOffset );
                context.lineTo( points[1].x + xOffset, points[1].y + yOffset );
                context.lineTo( points[2].x + xOffset, points[2].y + yOffset );
                context.lineTo( points[3].x + xOffset, points[3].y + yOffset );
                context.lineTo( points[4].x + xOffset, points[4].y + yOffset );
                context.lineTo( points[5].x + xOffset, points[5].y + yOffset );
                context.closePath();
                context.fill();
                context.stroke();
            }
        }
        
        drawShapes( redContext, 'rgba(255,0,0,0.7)', redSeed );
        drawShapes( greenContext, 'rgba(0,255,0,0.7)', greenSeed );
        drawShapes( blueContext, 'rgba(0,0,255,0.7)', blueSeed );
        
        var time = 0;
        
        // return step function
        return function( timeElapsed ) {
            time += timeElapsed;
            greenContext.clearRect( -0.6 * backgroundSize, -0.6 * backgroundSize, 1.2 * backgroundSize, 1.2 * backgroundSize );
            
            drawShapes( greenContext, 'rgba(0,255,0,0.7)', greenSeed, time );
        }
    };
    
})();

