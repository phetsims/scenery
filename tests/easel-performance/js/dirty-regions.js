
var phet = phet || {};
phet.tests = phet.tests || {};

(function(){
    
    phet.tests.sceneDirtyRegions = function( main ) {
        var scene = new phet.scene.Scene( main );
        var root = scene.root;
        root.layerType = phet.scene.layers.CanvasLayer;
        
        var sceneWidth = 1000;
        var sceneHeight = 500;
        var borderFactor = 6 / 5;
        
        var background = new phet.scene.Node();
        background.setShape( phet.scene.Shape.rectangle( -sceneWidth / 2 * borderFactor, -sceneHeight / 2 * borderFactor, sceneWidth * borderFactor, sceneHeight * borderFactor ) );
        background.fill = '#333333';
        background.stroke = '#000000';
        root.addChild( background );
        
        var nodes = new phet.scene.Node();
        root.addChild( nodes );
        
        for( var i = 0; i < 5000; i++ ) {
            var node = new phet.scene.Node();
            var radius = 10;
            
            // regular polygon
            node.setShape( phet.scene.Shape.regularPolygon( 6, radius ) );
            
            node.fill = phet.tests.themeColor( 0.5 );
            node.stroke = '#000000';
            
            node.setTranslation( ( Math.random() - 0.5 ) * sceneWidth, ( Math.random() - 0.5 ) * sceneHeight );
            
            nodes.addChild( node );
        }
        
        // center the root
        root.translate( main.width() / 2, main.height() / 2 );
        
        // generate the layers
        scene.rebuildLayers();
        
        // return step function
        return function( timeElapsed ) {
            // tweak a random node
            var node = nodes.children[_.random( 0, nodes.children.length - 1)];
            node.translate( ( Math.random() - 0.5 ) * 20, ( Math.random() - 0.5 ) * 20 );
            
            scene.updateScene();
        }
    };
    
})();

