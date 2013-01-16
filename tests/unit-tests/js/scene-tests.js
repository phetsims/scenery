

(function(){
    
    /*---------------------------------------------------------------------------*
    * TESTS BELOW
    *----------------------------------------------------------------------------*/        
    
    module( 'Canvas Scene tests' );
    
    test( 'Canvas 2D Context and Features', function() {
        var canvas = document.createElement( 'canvas' );
        var context = phet.canvas.initCanvas( canvas );
        
        ok( context, 'context' );
        
        var neededMethods = [
            'arc',
            'beginPath',
            'bezierCurveTo',
            'clearRect',
            'clip',
            'closePath',
            'fillStyle',
            'lineTo',
            'moveTo',
            'rect',
            'restore',
            'quadraticCurveTo',
            'save',
            'setTransform',
            'strokeStyle'
        ];
        _.each( neededMethods, function( method ) {
            ok( context[method] !== undefined, 'context.' + method );
        } );
    } );
    
    module( 'WebGL tests' );
    
    test( 'Canvas WebGL Context and Features', function() {
        var canvas = document.createElement( 'canvas' );
        var context = phet.webgl.initWebGL( canvas );
        ok( context, 'context' );
    } );
    
    // v5 canvas additions
    module( 'Bleeding Edge Canvas Support' );
    
    test( 'Canvas 2D v5 Features', function() {
        var canvas = document.createElement( 'canvas' );
        var context = phet.canvas.initCanvas( canvas );
        
        var neededMethods = [
            'ellipse'
        ];
        _.each( neededMethods, function( method ) {
            ok( context[method] !== undefined, 'context.' + method );
        } );
    } );
    
})();



