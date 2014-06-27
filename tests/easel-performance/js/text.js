
var phet = phet || {};
phet.tests = phet.tests || {};

(function(){
  "use strict";

  function buildBaseContext( main ) {
    var baseCanvas = document.createElement( 'canvas' );
    baseCanvas.id = 'base-canvas';
    baseCanvas.width = main.width();
    baseCanvas.height = main.height();
    main.append( baseCanvas );

    return baseCanvas.getContext( '2d' );
  }

  phet.tests.textBounds = function( main ) {
    var context = buildBaseContext( main );

    // for text testing: see http://www.whatwg.org/specs/web-apps/current-work/multipage/the-canvas-element.html#2dcontext
    // context: font, textAlign, textBaseline, direction
    // metrics: width, actualBoundingBoxLeft, actualBoundingBoxRight, etc.

    // consider something like http://mudcu.be/journal/2011/01/html5-typographic-metrics/

    var x = 10;
    var y = 100;
    // var str = "This is a test string";
    var str = "This is a test string \u222b \ufdfa \u00a7 \u00C1 \u00FF \u03A9 \u0906 \u79C1 \u9054 A\u030a\u0352\u0333\u0325\u0353\u035a\u035e\u035e 0\u0489 \u2588";

    context.font = '30px Arial';
    var metrics = context.measureText( str );
    context.fillStyle = '#ccc';
    context.fillRect( x - metrics.actualBoundingBoxLeft, y - metrics.actualBoundingBoxAscent,
      x + metrics.actualBoundingBoxRight, y + metrics.actualBoundingBoxDescent );
    context.fillStyle = '#000';
    context.fillText( str, x, y );

    var testTransform = dot.Matrix3.translation( 50, 150 ).timesMatrix( dot.Matrix3.scaling( 10, 0.1 ) );
    testTransform.canvasSetTransform( context );
    context.fillText( 'This is a test string', 0, 0 );

    // return step function
    return function( timeElapsed ) {

    }
  };

  phet.tests.textBoundTesting = function( main ) {
    // maybe getBoundingClientRect(), after appending with position: absolute top: 0, left: 0?
    // offsetWidth / offsetHeight? -- try positioning absolutely, left0top0, then also check offsetLeft, offsetTop -- similar to getBoundingClientRect()

    var scene = new scenery.Scene( main );

    var text = new scenery.Text( "Now with text!" );
    text.fill = '#000000';
    scene.addChild( text );

    // center the scene
    scene.translate( main.width() / 2, main.height() / 2 );

    // return step function
    return function( timeElapsed ) {
      text.rotate( timeElapsed );

      scene.updateScene();
    }
  };
})();
