
var phet = phet || {};
phet.tests = phet.tests || {};

// TODO:
// consider clearRect under transformed bounds. may be more optimal
// optimizations from http://www.html5rocks.com/en/tutorials/canvas/performance/

$(document).ready( function() {
    // the main div where all testing content is rendered
    var main = $('#main');
    var buttonRow = $('#buttonrow');
    
    // constants
    var activeClass = 'ui-btn-active';
    var boxSizeRatio = 0.75;
    var boxTotalSize = 200;
    
    // all of the individual test code and data is here
    var tests = [{
        testName: 'Boxes',
        testId: 'test-boxes',
        types: [{
            typeName: 'Easel 5',
            typeId: 'easel5',
            init: function( main ) {
                return phet.tests.easelVariableBox( main, 5 );
            }
        },{
            typeName: 'Custom 5',
            typeId: 'custom5',
            init: function( main ) {
                return phet.tests.customVariableBox( main, 5 );
            }
        },{
            typeName: 'Scene 5',
            typeId: 'scene5',
            init: function( main ) {
                return phet.tests.sceneVariableBox( main, 5 );
            }
        },{
            typeName: 'SVG 5',
            typeId: 'svg5',
            init: function( main ) {
                return phet.tests.svgVariableBox( main, 5 );
            }
        },{
            typeName: 'Easel 50',
            typeId: 'easel50',
            init: function( main ) {
                return phet.tests.easelVariableBox( main, 50 );
            }
        },{
            typeName: 'Custom 50',
            typeId: 'custom50',
            init: function( main ) {
                return phet.tests.customVariableBox( main, 50 );
            }
        },{
            typeName: 'Scene 50',
            typeId: 'scene50',
            init: function( main ) {
                return phet.tests.sceneVariableBox( main, 50 );
            }
        },{
            typeName: 'SVG 50',
            typeId: 'svg50',
            init: function( main ) {
                return phet.tests.svgVariableBox( main, 50 );
            }
        }]
    },{
        testName: 'Misc',
        testId: 'test-misc',
        types: [{
            typeName: '50x Canvas Creation',
            typeId: 'canvasCreation',
            init: function( main ) {
                return function( timeElapsed ) {
                    for( var i = 0; i < 50; i++ ) {
                        main.empty();
                        
                        var canvas = document.createElement( 'canvas' );
                        canvas.width = main.width();
                        canvas.height = main.height();
                        main.append( canvas );
                        
                        var context = phet.canvas.initCanvas( canvas );
                    }
                };
            }
        },{
            typeName: 'Text Bounds',
            typeId: 'boundsTest',
            init: function( main ) {
                return phet.tests.textBounds( main )
            }
        },{
            typeName: 'Winding', // pure black and lighter blue will appear if the fillRule property exists in the Canvas 2d context
            typeId: 'winding',
            init: function( main ) {
                var baseCanvas = document.createElement( 'canvas' );
                baseCanvas.id = 'base-canvas';
                baseCanvas.width = main.width();
                baseCanvas.height = main.height();
                main.append( baseCanvas );
                
                var context = phet.canvas.initCanvas( baseCanvas );
                
                var x = baseCanvas.width / 2;
                var y = baseCanvas.height / 2;
                
                context.fillStyle = '#666666';
                var canvasPadding = 30;
                context.fillRect( canvasPadding, canvasPadding, baseCanvas.width - 2 * canvasPadding, baseCanvas.height - 2 * canvasPadding );
                
                function drawPattern( callbackAfterClose ) {
                    context.beginPath();
                    context.arc( x + 50, y, 100, 0, 2 * Math.PI, true );
                    context.arc( x + 100, y, 25, 0, 2 * Math.PI, true );
                    context.closePath();
                    if( callbackAfterClose ) { callbackAfterClose( '#FF0000' ); }
                    
                    context.arc( x - 50, y, 100, 2 * Math.PI, 0, false );
                    context.arc( x - 100, y, 25, 0, 2 * Math.PI, true );
                    context.closePath();
                    if( callbackAfterClose ) { callbackAfterClose( '#FFFF00' ); }
                    
                    context.arc( x + 200, y, 100, 2 * Math.PI, 0, false );
                    context.closePath();
                    if( callbackAfterClose ) { callbackAfterClose( '#00FF00' ); }
                    
                    context.arc( x - 200, y, 100, 0, 2 * Math.PI, true );
                    context.closePath();
                    if( callbackAfterClose ) { callbackAfterClose( '#FF0000' ); }
                    
                    context.arc( x, y - 100, 100, 0, 2 * Math.PI, true );
                    context.closePath();
                    if( callbackAfterClose ) { callbackAfterClose( '#FF0000' ); }
                    
                    context.arc( x, y - 100, 50, 2 * Math.PI, 0, false );
                    context.closePath();
                    if( callbackAfterClose ) { callbackAfterClose( '#00FF00' ); }
                    
                    context.arc( x, y + 100, 100, 2 * Math.PI, 0, false );
                    context.closePath();
                    if( callbackAfterClose ) { callbackAfterClose( '#00FF00' ); }
                    
                    context.arc( x, y + 100, 50, 2 * Math.PI, 0, false );
                    context.closePath();
                    if( callbackAfterClose ) { callbackAfterClose( '#00FF00' ); }
                }
                
                drawPattern();
                context.fillRule = 'nonzero';
                context.fillStyle = '#000000';
                context.fill();
                
                drawPattern();
                context.fillRule = 'evenodd';
                context.fillStyle = 'rgba( 0, 0, 255, 0.25 )';
                context.fill();
                
                drawPattern( function( style ) {
                    context.strokeStyle = style;
                    context.stroke();
                    context.beginPath();
                } );
                
                return function( timeElapsed ) {
                    
                }
            }
        },{
            typeName: 'Clip',
            typeId: 'clipTest',
            init: function( main ) {
                var baseCanvas = document.createElement( 'canvas' );
                baseCanvas.id = 'base-canvas';
                baseCanvas.width = main.width();
                baseCanvas.height = main.height();
                main.append( baseCanvas );
                
                var context = phet.canvas.initCanvas( baseCanvas );
                
                var x = baseCanvas.width / 2;
                var y = baseCanvas.height / 2;
                
                var timer = { total: 0 };
                
                return function( timeElapsed ) {
                    timer.total += timeElapsed;
                    
                    context.save();
                    
                    // context.resetTransform(); // TODO: why is this not working?
                    context.setTransform( 1, 0, 0, 1, 0, 0 );
                    context.clearRect( 0, 0, baseCanvas.width, baseCanvas.height );
                    
                    context.setTransform( 1, 0, 0, 1, x - 100, y - 100 );
                    
                    context.beginPath();
                    context.rect( 90, 0, 20, 200 );
                    context.clip();
                    
                    context.transform( 1, 0, 0, 1, Math.sin( timer.total ) * 100, 0 );
                    
                    context.beginPath();
                    context.arc( 200, 200, 200, 0, 2 * Math.PI, true );
                    context.clip();
                    context.beginPath();
                    context.arc( 0, 0, 200, 0, 2 * Math.PI, true );
                    context.clip();
                    context.beginPath();
                    context.rect( 0, 0, 200, 200 );
                    context.fillStyle = '#000000';
                    context.fill();
                    
                    context.restore();
                };
            }
        }]
    },{
        testName: 'Layers',
        testId: 'test-layers',
        types: [{
            typeName: 'Demo',
            typeId: 'demo',
            init: function( main ) {
                return phet.tests.layeringTests( main );
            }
        }]
    },{
        testName: 'Placebo',
        testId: 'test-placebo',
        types: [{
            typeName: 'FPS "Control": empty step function',
            typeId: 'nothingDone',
            init: function( main ) {
                return function( timeElapsed ) {};
            }
        }]
    }];
    
    function buildEaselStage() {
        var canvas = document.createElement( 'canvas' );
        canvas.id = 'easel-canvas';
        canvas.width = main.width();
        canvas.height = main.height();
        main.append( canvas );

        return new createjs.Stage( canvas );
    }
    
    function buildBaseContext() {
        var baseCanvas = document.createElement( 'canvas' );
        baseCanvas.id = 'base-canvas';
        baseCanvas.width = main.width();
        baseCanvas.height = main.height();
        main.append( baseCanvas );
        
        return phet.canvas.initCanvas( baseCanvas );
    }
    
    // var currentTest = tests[0];
    // var currentType = tests[0].types[0];
    var currentTest = tests[2];
    var currentType = tests[2].types[0];
    
    function createButtonGroup() {
        var result = $( document.createElement( 'span' ) );
        result.attr( 'data-role', 'controlgroup' );
        result.attr( 'data-type', 'horizontal' );
        result.attr( 'data-mini', 'true' );
        result.css( 'padding-right', '20px' );   
        return result;
    }
    
    function createButton( title ) {
        var result = $( document.createElement( 'a' ) );
        result.attr( 'data-role', 'button' );
        result.attr( 'href', '#' );
        result.text( title );
        return result;
    }
    
    function updateHighlights() {
        // update the button highlights
        _.each( tests, function( otherTest ) {
            if( otherTest == currentTest ) {
                // select this button as active
                $( '#' + otherTest.testId ).addClass( activeClass );
                
                // display its types
                $( '#types-' + otherTest.testId ).css( 'display', 'inline' );
                
                // set the type selected
                _.each( otherTest.types, function( otherType ) {
                    if( otherType == currentType ) {
                        $( '#' + otherTest.testId + '-' + otherType.typeId ).addClass( activeClass );
                    } else {
                        $( '#' + otherTest.testId + '-' + otherType.typeId ).removeClass( activeClass );
                    }
                } );
            } else {
                // and set others as inactive
                $( '#' + otherTest.testId ).removeClass( activeClass );
                
                // hide their types
                $( '#types-' + otherTest.testId ).css( 'display', 'none' );
            }
        } );
    }
    
    // first group of buttons, one for each major test
    var testButtons = createButtonGroup();
    testButtons.attr( 'id', 'test-buttons' );
    buttonRow.append( testButtons );
    
    _.each( tests, function( test ) {
        // make a button for each test
        var testButton = createButton( test.testName );
        testButton.attr( 'id', test.testId );
        testButtons.append( testButton );
        
        testButton.on( 'click', function( event, ui ) {
            currentTest = test;
            currentType = test.types[0];
            
            updateHighlights();            
            changeTest();            
        } );
        
        // group of buttons to the right for each test
        var typeButtons = createButtonGroup();
        typeButtons.attr( 'id', 'types-' + test.testId );
        buttonRow.append( typeButtons );
        
        _.each( test.types, function( type ) {
            // add a type button for each type
            var typeButton = createButton( type.typeName );
            typeButton.attr( 'id', test.testId + '-' + type.typeId );
            typeButtons.append( typeButton );
            
            typeButton.on( 'click', function( event, ui ) {
                currentType = type;
                
                updateHighlights();
                changeTest();
            } );
        } );
        
        // don't show the type buttons at the start
        typeButtons.css( 'display', 'none' );
    } );
    
    buttonRow.trigger('create');
    
    updateHighlights();
    
    // closure used over this for running steps
    var step = function( timeElapsed ) {
        
    };
    
    // called whenever the test is supposed to change
    function changeTest() {
        // clear the main area
        main.empty();
        
        // run the initialization and change the step function
        step = currentType.init( main );
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

} );
