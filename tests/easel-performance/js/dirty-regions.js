
var phet = phet || {};
phet.tests = phet.tests || {};

(function(){
    
    phet.tests.sceneDirtyRegions = function( main ) {
        var scene = new phet.scene.Scene( main );
        var root = scene.root;
        root.layerType = phet.scene.layers.CanvasLayer;
        
        var background = new phet.scene.Node();
        background.setShape( phet.scene.Shape.rectangle( -600, -300, 1200, 600 ) );
        background.fill = '#333333';
        background.stroke = '#000000';
        root.addChild( background );
        
        for( var i = 0; i < 5000; i++ ) {
            var node = new phet.scene.Node();
            var radius = 10;
            
            // regular polygon
            node.setShape( phet.scene.Shape.regularPolygon( 6, radius ) );
            
            node.fill = phet.tests.themeColor( 0.5 );
            node.stroke = '#000000';
            
            node.translate( ( Math.random() - 0.5 ) * 1000, ( Math.random() - 0.5 ) * 500 );
            
            root.addChild( node );
        }
        
        // center the root
        root.translate( main.width() / 2, main.height() / 2 );
        
        // generate the layers
        scene.rebuildLayers();
        
        scene.updateScene();
        
        // return step function
        return function( timeElapsed ) {
            // var oldBounds = root.getBounds();
            
            // greens.rotate( timeElapsed );
            
            // root.validateBounds();
            
            // var newBounds = root.getBounds();
            // var combinedBounds = oldBounds.union( newBounds ).dilated( 1 );
            
            // if( useLayers ) {
            //     greens._layerBeforeRender.prepareBounds( combinedBounds );
            // } else {
            //     _.each( scene.layers, function( layer ) {
            //         layer.prepareBounds( combinedBounds );
            //     } );
            // }
            
            // scene.updateScene();
        }
    };
    
})();

