
var phet = phet || {};
phet.tests = phet.tests || {};

// TODO:
// consider clearRect under transformed bounds. may be more optimal
// optimizations from http://www.html5rocks.com/en/tutorials/canvas/performance/

$(document).ready( function() {
    "use strict";
    
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
            typeName: 'Scene Text',
            typeId: 'texty',
            init: function( main ) {
                return phet.tests.textBoundTesting( main )
            }
        },{
            typeName: 'Bezier',
            typeId: 'bezier',
            init: function( main ) {
                // TODO: cusps at (100, 100), (300, 200), (200, 200), (200, 100) for cubic
                var scene = new phet.scene.Scene( main );
                
                var mainCurve = new phet.scene.Shape.Segment.Quadratic(
                    new phet.math.Vector2( 100, 100 ),
                    new phet.math.Vector2( 230, 100 ),
                    new phet.math.Vector2( 150, 350 )
                );
                
                // TODO: a way to pass an array and run commands without explicitly adding pieces?
                // scene.root.addChild( new phet.scene.Node( {
                //     shape: new phet.scene.Shape( [
                //         new phet.scene.Shape.Piece.MoveTo( mainCurve.start ),
                //         new phet.scene.Shape.Piece.LineTo( mainCurve.control ),
                //         new phet.scene.Shape.Piece.LineTo( mainCurve.end )
                //     ] ),
                //     stroke: '#ff0000'
                // } ) );
                
                scene.root.addChild( new phet.scene.Node( {
                    shape: new phet.scene.Shape( [
                        new phet.scene.Shape.Piece.MoveTo( mainCurve.start ),
                        new phet.scene.Shape.Piece.QuadraticCurveTo( mainCurve.control, mainCurve.end )
                    ] ),
                    stroke: '#000000'
                } ) );
                
                _.each( [ -55, -40, -30, -15, -5, 5, 15, 30, 40, 55 ], function( offset ) {
                    var offsetPieces = mainCurve.offsetTo( offset, true );
                    
                    scene.root.addChild( new phet.scene.Node( {
                        shape: new phet.scene.Shape( offsetPieces ),
                        stroke: '#0000ff'
                    } ) );
                } );
                
                return function( timeElapsed ) {
                    scene.updateScene();
                };
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
        },{
            typeName: 'Images',
            typeId: 'images',
            init: function( main ) {
                var scene = new phet.scene.Scene( main );
                
                var imageSource = document.createElement( 'img' );
                imageSource.onload = function( e ) {
                    var image = new phet.scene.Image( imageSource );
                    image.translate( -image.getSelfBounds().width() / 2, -image.getSelfBounds().height() / 2 );
                    scene.root.addChild( image );
                };
                imageSource.src = 'http://phet.colorado.edu/images/phet-logo.gif';
                
                // center it
                scene.root.translate( main.width() / 2, main.height() / 2 );
                
                return function( timeElapsed ) {
                    scene.root.rotate( timeElapsed );
                    scene.updateScene();
                };
            }
        },{
            typeName: 'DOM',
            typeId: 'dom',
            init: function( main ) {
                var scene = new phet.scene.Scene( main );
                
                var element = document.createElement( 'iframe' );
                $( element ).attr( 'width', '560' );
                $( element ).attr( 'height', '315' );
                $( element ).attr( 'src', 'http://www.youtube.com/embed/N17IM7LspU8' );
                $( element ).attr( 'frameborder', '0' );
                
                // var unclickableForm = document.createElement( 'form' );
                // unclickableForm.innerHTML = 'Unclickable<br>First name: <input type="text" name="firstname"><br>Last name: <input type="text" name="lastname">';
                
                var clickableForm = document.createElement( 'form' );
                clickableForm.innerHTML = 'Clickable<br>First name: <input type="text" name="firstname"><br>Last name: <input type="text" name="lastname">';
                
                var bigForm = document.createElement( 'form' );
                bigForm.innerHTML = 'And now scalable!<br><input type="text" name="stuff">';
                
                var container = new phet.scene.Node();
                container.setLayerType( phet.scene.DOMLayer );
                
                var videoNode = new phet.scene.DOM( element );
                videoNode.translate( -560 / 2, -315 / 2 );
                videoNode.interactive = true;
                container.addChild( videoNode );
                
                // var unclickableFormNode = new phet.scene.DOM( unclickableForm );
                // unclickableFormNode.translate( 0, 160 );
                // container.addChild( unclickableFormNode );
                
                var clickableFormNode = new phet.scene.DOM( clickableForm );
                clickableFormNode.translate( -560 / 2, 160 );
                clickableFormNode.interactive = true;
                container.addChild( clickableFormNode );
                
                var bigFormNode = new phet.scene.DOM( bigForm );
                bigFormNode.translate( 0, 160 );
                bigFormNode.scale( 4 );
                bigFormNode.interactive = true;
                container.addChild( bigFormNode );
                
                scene.root.addChild( container );
                
                // var background = new phet.scene.Node();
                // background.setShape( phet.scene.Shape.rectangle( -400, -400, 800, 800 ) );
                // background.setFill( 'rgba(230,255,230,0.5)' );
                // scene.root.addChild( background );
                
                // center it
                scene.root.translate( main.width() / 2, main.height() / 2 );
                
                return function( timeElapsed ) {
                    scene.root.rotate( timeElapsed / 3 );
                    
                    // TODO: get updateScene to work with this
                    scene.renderScene();
                };
            }
        },{
            typeName: 'Strokes',
            typeId: 'strokes',
            init: function( main ) {
                var baseCanvas = document.createElement( 'canvas' );
                baseCanvas.id = 'base-canvas';
                baseCanvas.width = main.width();
                baseCanvas.height = main.height();
                main.append( baseCanvas );
                
                var context = phet.canvas.initCanvas( baseCanvas );
                
                var angle = 0;
                
                return function( timeElapsed ) {
                    angle += timeElapsed;
                    
                    context.clearRect( 0, 0, baseCanvas.width, baseCanvas.height );
                    
                    context.lineWidth = 20;
                    
                    function example( join, x ) {
                        context.beginPath();
                        context.moveTo( x, 100 );
                        context.lineTo( x, 150 );
                        context.lineTo( x + Math.cos( angle ) * 50, 150 + Math.sin( angle ) * 50 );
                        context.lineJoin = join;
                        context.stroke();
                    }
                    
                    example( 'miter', 100 );
                    example( 'bevel', 250 );
                    example( 'round', 400 );
                    
                };
            }
        },{
            // should bog down in non-production versions
            typeName: 'Debug Speed',
            typeId: 'debugSpeed',
            init: function( main ) {
                return function( timeElapsed ) {
                    for( var i = 0; i < 100; i++ ) {
                        phet.debugAssert( function() {
                            var array = [];
                            for( var i = 0; i < 100; i++ ) {
                                array.push( Math.random() );
                            }
                            for( var k = 0; k < 500; k++ ) {
                                _.shuffle( array );
                            }
                            return true;
                        }, 'Message Never Seen' );
                    }
                };
            }
        }]
    },{
        testName: 'Layers',
        testId: 'test-layers',
        types: [{
            typeName: 'Scene With',
            typeId: 'sceneWith',
            init: function( main ) {
                return phet.tests.sceneLayeringTests( main, true );
            }
        },{
            typeName: 'Scene Without',
            typeId: 'sceneWithout',
            init: function( main ) {
                return phet.tests.sceneLayeringTests( main, false );
            }
        },{
            typeName: 'Easel',
            typeId: 'easel',
            init: function( main ) {
                return phet.tests.easelLayeringTests( main, false );
            }
        },{
            typeName: 'Custom',
            typeId: 'custom',
            init: function( main ) {
                return phet.tests.customLayeringTests( main, false );
            }
        }]
    },{
        testName: 'Dirty',
        testId: 'test-dirty',
        types: [{
            typeName: 'Scene 1',
            typeId: 'scene1',
            init: function( main ) {
                return phet.tests.sceneDirtyRegions( main, 1 );
            }
        },{
            typeName: 'Scene 50',
            typeId: 'Scene50',
            init: function( main ) {
                return phet.tests.sceneDirtyRegions( main, 50 );
            }
        },{
            typeName: 'Scene All',
            typeId: 'Sceneall',
            init: function( main ) {
                return phet.tests.sceneDirtyRegions( main, 0 );
            }
        },{
            typeName: 'Easel 1',
            typeId: 'easel1',
            init: function( main ) {
                return phet.tests.easelDirtyRegions( main, 1 );
            }
        },{
            typeName: 'Easel 50',
            typeId: 'easel50',
            init: function( main ) {
                return phet.tests.easelDirtyRegions( main, 50 );
            }
        },{
            typeName: 'Easel All',
            typeId: 'easelall',
            init: function( main ) {
                return phet.tests.easelDirtyRegions( main, 0 );
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
        var fpsCount = 20;
        var fpsIndex = 0;
        // stuffs timestamps in a round-robin fashion into here
        var timeEntries = _.map( _.range( fpsCount ), function( a ) { return 0; } );
        
        // stored, so we can reset the FPS meter on a change
        var lastType = null;
        
        // only change the readout so often
        var msReadoutDelay = 250;
        var lastTimeReadoutUpdated = 0;
        
        function updateFps() {
            // wipe the meter until we have all fresh readings whenever a test's type changes
            if( lastType != currentType ) {
                timeEntries = _.map( timeEntries, function( a ) { return 0; } );
                lastType = currentType;
            }
            
            var timestamp = Date.now();
            timeEntries[ fpsIndex ] = timestamp;
            fpsIndex = ( fpsIndex + 1 ) % fpsCount;
            var timeChange = ( timestamp - timeEntries[ fpsIndex ] ) / 1000;
            var fps = fpsCount / timeChange;
            
            if( fps < 1 || timestamp - lastTimeReadoutUpdated > msReadoutDelay ) {
                lastTimeReadoutUpdated = timestamp;
                if( fps < 1 ) {
                    fpsReadout.text( "-" );
                } else {
                    fpsReadout.text( fps.toFixed( 1 ) );
                }
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
    
    
    /*---------------------------------------------------------------------------*
    * other miscellaneous functions used for testing
    *----------------------------------------------------------------------------*/        
    
    phet.tests.themeColor = function( alpha, blend ) {
        var scale = Math.floor( 511 * Math.random() ) - 255;

        var red = 255;
        var green = ( scale > 0 ? scale : 0 );
        var blue = ( scale < 0 ? -scale : 0 );
        
        function combine( main, backup ) {
            return Math.floor( main * ( 1 - blend ) + backup * blend );
        }

        if( alpha !== undefined ) {
            if( blend !== undefined ) {
                return "rgba(" + combine( red, green ) + "," + combine( green, blue ) + "," + combine( blue, red ) + "," + alpha + ")";
            } else {
                return "rgba(" + red + "," + green + "," + blue + "," + alpha + ")";
            }
        } else {
            return "rgb(" + red + "," + green + "," + blue + ")";
        }
    };
} );
