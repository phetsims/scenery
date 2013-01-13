
var phet = phet || {};
phet.tests = phet.tests || {};

(function(){
    
    var Piece = phet.scene.Shape.Piece;
    
    var backgroundSize = 300;
    var count = 500;
    
    phet.tests.sceneLayeringTests = function( main, useLayers ) {
        var radius = 15;
        var scene = new phet.scene.Scene( main );
        var root = scene.root;
        root.layerType = phet.scene.layers.CanvasLayer;
        
        function buildShapes( color ) {
            var background = new phet.scene.Node();
            root.addChild( background );
            if( useLayers ) {
                background.layerType = phet.scene.layers.CanvasLayer;
            }
            for( var i = 0; i < count; i++ ) {
                var node = new phet.scene.Node();
                var radius = 10;
                
                // regular polygon
                node.setShape( phet.scene.Shape.regularPolygon( 6, radius ) );
                
                node.fill = color;
                node.stroke = '#000000';
                
                node.translate( ( Math.random() - 0.5 ) * backgroundSize, ( Math.random() - 0.5 ) * backgroundSize );
                
                background.addChild( node );
            }
            return background;
        }
        
        var reds = buildShapes( 'rgba(255,0,0,0.7)' );
        var greens = buildShapes( 'rgba(0,255,0,0.7)' );
        var blues = buildShapes( 'rgba(0,0,255,0.7)' );
        
        // center the root
        root.translate( main.width() / 2, main.height() / 2 );
        
        // generate the layers
        scene.rebuildLayers();
        
        // return step function
        return function( timeElapsed ) {
            var oldBounds = root.getBounds();
            
            greens.rotate( timeElapsed );
            
            root.validateBounds();
            var newBounds = root.getBounds();
            var combinedBounds = oldBounds.union( newBounds ).dilated( 1 );
            
            if( useLayers ) {
                greens._layerBeforeRender.prepareBounds( combinedBounds );
            } else {
                _.each( scene.layers, function( layer ) {
                    layer.prepareBounds( combinedBounds );
                } );
            }
            
            scene.updateScene();
        }
    };
    
    phet.tests.easelLayeringTests = function( main ) {
        var radius = 15;
        
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
    }
    
})();

