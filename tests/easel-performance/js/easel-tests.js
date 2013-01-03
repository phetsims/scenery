
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
            typeName: 'Easel 100',
            typeId: 'easel100',
            init: function( main ) {
                return phet.tests.easelVariableBox( main, 100 );
            }
        },{
            typeName: 'Custom 100',
            typeId: 'custom100',
            init: function( main ) {
                return phet.tests.customVariableBox( main, 100 );
            }
        },{
            typeName: 'Scene 100',
            typeId: 'scene100',
            init: function( main ) {
                return phet.tests.sceneVariableBox( main, 100 );
            }
        }]
    },{
        testName: 'Text',
        testId: 'test-text',
        types: [{
            typeName: 'Bounds Test',
            typeId: 'boundsTest',
            init: function( main ) {
                return phet.tests.textBounds( main )
            }
        }]
    },{
        testName: 'Placebo',
        testId: 'test-placebo',
        types: [{
            typeName: 'Nothing Done',
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
    
    var currentTest = tests[0];
    var currentType = tests[0].types[0];
    
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
