
var phet = phet || {};
phet.tests = phet.tests || {};

(function(){
    
    var Piece = phet.scene.Shape.Piece;
    
    var backgroundSize = 300;
    
    phet.tests.layeringTests = function( main, useLayers ) {
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
            for( var i = 0; i < 500; i++ ) {
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
    
})();

