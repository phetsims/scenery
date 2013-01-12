
var phet = phet || {};
phet.tests = phet.tests || {};

(function(){
    
    var Piece = phet.scene.Shape.Piece;
    
    var backgroundSize = 500;
    
    phet.tests.layeringTests = function( main ) {
        var scene = new phet.scene.Scene( main );
        
        var root = scene.root;
        
        root.layerType = phet.scene.layers.CanvasLayer;
        
        var background = new phet.scene.Node();
        root.addChild( background );
        for( var i = 0; i < 1000; i++ ) {
            var node = new phet.scene.Node();
            var n = 6;
            var radius = 15;
            
            // regular polygon
            node.setShape( new phet.scene.Shape( _.map( _.range( n ), function( k ) {
                var theta = 2 * Math.PI * k / n;
                return Piece.lineTo( radius * Math.cos( theta ), radius * Math.sin( theta ) );
            } ), true ) );
            
            node.fill = 'rgba(255,255,255,0.8)';
            node.stroke = '#000000';
            
            node.translate( ( Math.random() - 0.5 ) * 500, ( Math.random() - 0.5 ) * 500 );
            
            background.addChild( node );
        }
        
        // center the root
        root.translate( main.width() / 2, main.height() / 2 );
        
        // generate the layers
        scene.rebuildLayers();
        
        // return step function
        return function( timeElapsed ) {
            var bounds = root.getBounds();
            // clear around another pixel or so, for antialiasing!
            root._layerBeforeRender.context.clearRect( bounds.x() - 1, bounds.y() - 1, bounds.width() + 2, bounds.height() + 2 );
            
            scene.renderScene();
        }
    };
    
})();

