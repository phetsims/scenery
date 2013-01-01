
var phet = phet || {};
phet.tests = phet.tests || {};

$(document).ready( function() {
    // constants
    var activeClass = 'ui-btn-active';
    
    var type = "type-easel";
    var test = "test-boxes";
    
    // hook up activity changes so we can toggle sets of buttons
    _.each( [
        [ 'type-easel', 'type-custom' ], // Easel / Custom
        [ 'test-boxes', 'test-text' ] // Separate tests
    ], function( list ) {
        _.each( list, function( selector ) {
            // set up initial buttons
            if( selector == type || selector == test ) {
                $( '#' + selector ).addClass( activeClass );
            }
            
            // handle toggling for each button
            $( '#' + selector ).on( 'click', function( event, ui ) {
                // update the button highlights
                _.each( list, function( button ) {
                    if( button == selector ) {
                        // select this button as active
                        $( '#' + button ).addClass( activeClass );
                    } else {
                        // and set others as inactive
                        $( '#' + button ).removeClass( activeClass );
                    }
                } );
                
                // change the test
                if( selector.substring( 0, 4 ) == "type" ) {
                    type = selector;
                } else {
                    test = selector;
                }
                changeTest();
            } );
        } );
    } );
    
    var main = $('#main');
    
    // closure used over this for running steps
    var step = function( timeElapsed ) {
        
    };
    
    // called whenever the test is supposed to change
    function changeTest() {
        // clear the main area
        main.empty();
        
        if( type == 'type-easel' ) {
            var canvas = document.createElement( 'canvas' );
            canvas.id = 'easel-canvas';
            canvas.width = main.width();
            canvas.height = main.height();
            main.append( canvas );

            var stage = new createjs.Stage(canvas);
            
            if( test == 'test-boxes' ) {
                var shape = new createjs.Shape();
                shape.graphics.beginFill('rgba(255,0,0,1)').drawRect( 0, 0, 100, 100 );
                shape.x = main.width() / 2;
                shape.y = main.height() / 2;
                stage.addChild(shape);
                
                step = function( timeElapsed ) {
                    shape.rotation += timeElapsed * 180 / Math.PI;
                    stage.update();
                };
            } else if( test == 'test-text' ) {
                var text = new createjs.Text( 'Test String', '20px Arial', '#555555' );
                stage.addChild( text );
                text.x = 100;
                text.y = 100;
                
                step = function( timeElapsed ) {
                    text.rotation += timeElapsed * 180 / Math.PI;
                    stage.update();
                };
            }
        } else {
            var baseCanvas = document.createElement( 'canvas' );
            baseCanvas.id = 'base-canvas';
            baseCanvas.width = main.width();
            baseCanvas.height = main.height();
            main.append( baseCanvas );
            var baseContext = phet.canvas.initCanvas( baseCanvas );
            
            if( test == 'test-boxes' ) {
                var x = main.width() / 2;
                var y = main.height() / 2;
                var rotation = 0;
                baseContext.fillStyle = 'rgba(255,0,0,1)';
                step = function( timeElapsed ) {
                    // clear old location
                    baseContext.clearRect( x - 150, y - 150, 300, 300 );
                    // TODO: consider whether clearRect is faster if it's not under a transform!
                    
                    baseContext.save();
                    rotation += timeElapsed;
                    baseContext.translate( x, y );
                    baseContext.rotate( rotation );
                    
                    baseContext.beginPath();
                    baseContext.moveTo( 0, 0 );
                    baseContext.lineTo( 100, 0 );
                    baseContext.lineTo( 100, 100 );
                    baseContext.lineTo( 0, 100 );
                    baseContext.closePath();
                    baseContext.fill();
                    
                    baseContext.restore();
                };
            } else if( test == 'test-text' ) {
                step = function( timeElapsed ) {
                    
                };
            }
        }
    }
    
    // set up averaging FPS meter and calling of the step() function
    (function(){
        // records the last #fpsCount timestamps, and displays the average FPS over that time
        var fpsReadout = $('#fps-readout');
        var fpsCount = 10;
        var fpsIndex = 0;
        // stuffs timestamps in a round-robin fashion into here
        var timeEntries = _.map( _.range( fpsCount ), function( a ) { return 0; } );
        function updateFps() {
            var timestamp = Date.now();
            timeEntries[ fpsIndex ] = timestamp;
            fpsIndex = ( fpsIndex + 1 ) % fpsCount;
            var timeChange = ( timestamp - timeEntries[ fpsIndex ] ) / 1000;
            var fps = 10 / timeChange;
            if( fps < 1 ) {
                fpsReadout.text( "-" );
            } else {
                fpsReadout.text( fps.toFixed( 1 ) );
            }
        }
        
        var lastTime = 0;
        var timeElapsed = 0;
        
        // setting up regular calls for step()
        function tick() {
            window.requestAnimationFrame( tick, main[0] );
            var timeNow = new Date().getTime();
            if ( lastTime != 0 ) {
                timeElapsed = (timeNow - lastTime) / 1000.0;
            }
            lastTime = timeNow;
            
            if( timeElapsed != 0 ) {
                step( timeElapsed );
            }
            updateFps();
        }
        window.requestAnimationFrame( tick, main[0] );
    })();
    
    // handle window resizing
    var resizer = function () {
        main.width( window.innerWidth );
        main.height( window.innerHeight );
        changeTest();
    };
    $( window ).resize( resizer );
    resizer();
    
    // for text testing: see http://www.whatwg.org/specs/web-apps/current-work/multipage/the-canvas-element.html#2dcontext
    // context: font, textAlign, textBaseline, direction
    // metrics: width, actualBoundingBoxLeft, actualBoundingBoxRight, etc.
} );
