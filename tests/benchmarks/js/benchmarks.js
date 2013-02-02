
(function(){

    function loadScript( src, callback ) {
        var called = false;

        var script = document.createElement( 'script' );
        script.type = 'text/javascript';
        script.async = true;
        script.onload = script.onreadystatechange = function() {
            var state = this.readyState;
            if ( state && state != "complete" && state != "loaded" ) {
                return;
            }

            if( !called ) {
                called = true;
                callback();
            }
        };
        script.src = src;

        var other = document.getElementsByTagName( 'script' )[0];
        other.parentNode.insertBefore( script, other );
    }


    loadScript( '../../phet-scene-min.js', function() {
        console.log( new phet.scene.Node() );
        loadScript( 'js/current.js', function() {} );
    } );

    // var resultDiv = $( '#profiling-results' );
    // var suite = new Benchmark.Suite;
    // var main = $( '#main' );
    // suite.add( 'canvas/context creation', function() {
    //     var canvas = document.createElement( 'canvas' );
    //     var context = phet.canvas.initCanvas( canvas );
    // } ).add( 'simple scene creation and iteration', function() {
    //     var scene = new phet.scene.Scene( $( '#main' ) );
    //     var root = scene.root;

    //     scene.updateScene();
    // }, {
    //     'onCycle': function() {
    //         main.empty();
    //     }
    // } ).on( 'cycle', function( event ) {
    //     console.log( event );
    //     resultDiv.text( resultDiv.text() + " " + String( event ) );
    // } ).run();
})();
