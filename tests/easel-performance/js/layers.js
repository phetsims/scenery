
var phet = phet || {};
phet.tests = phet.tests || {};

(function(){
    
    var Piece = phet.scene.Shape.Piece;
    
    var backgroundSize = 500;
    
    phet.tests.layeringTests = function( main ) {
        var radius = 15;
        var scene = new phet.scene.Scene( main );
        var root = scene.root;
        root.layerType = phet.scene.layers.CanvasLayer;
        
        _.each( ['rgba(255,0,0,0.7)', 'rgba(0,255,0,0.7)', 'rgba(0,0,255,0.7)'], function( color ) {
            var background = new phet.scene.Node();
            root.addChild( background );
            background.layerType = phet.scene.layers.CanvasLayer;
            for( var i = 0; i < 500; i++ ) {
                var node = new phet.scene.Node();
                var radius = 15;
                
                // regular polygon
                node.setShape( phet.scene.Shape.regularPolygon( 6, radius ) );
                
                node.fill = color;
                node.stroke = '#000000';
                
                node.translate( ( Math.random() - 0.5 ) * 500, ( Math.random() - 0.5 ) * 500 );
                
                background.addChild( node );
            }
        } );
        
        // center the root
        root.translate( main.width() / 2, main.height() / 2 );
        
        // generate the layers
        scene.rebuildLayers();
        
        // return step function
        return function( timeElapsed ) {
            var bounds = root.getBounds();
            // clear around another pixel or so, for antialiasing!
            _.each( scene.layers, function( layer ) {
                // TODO: dead regions!
                layer.context.clearRect( bounds.x() - 1, bounds.y() - 1, bounds.width() + 2, bounds.height() + 2 );
            } );
            
            scene.renderScene();
        }
    };
    
})();

