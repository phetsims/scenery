
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
        
        // var pickedNode = background.children[ _.random( background.children.length - 1 ) ];
        // pickedNode.fill = 'rgba(255,0,0,1)';
        // pickedNode.layerType = phet.scene.layers.CanvasLayer;
        
        // generate the layers
        scene.rebuildLayers();
        
        // TODO: remove after debugging
        for( var i = 0; i < scene.layers.length; i++ ) {
            scene.layers[i].z = i;
        }
        window.f = function(boo) { return [boo._layerBeforeRender ? boo._layerBeforeRender.z : '-',boo._layerAfterRender ? boo._layerAfterRender.z : '-']; }
        window.scene = scene;
        window.root = root;
        
        // return step function
        return function( timeElapsed ) {
            var bounds = root.getBounds();
            // clear around another pixel or so, for antialiasing!
            root._layerBeforeRender.context.clearRect( bounds.x() - 1, bounds.y() - 1, bounds.width() + 2, bounds.height() + 2 );
            
            scene.renderScene();
        }
    };
    
})();

