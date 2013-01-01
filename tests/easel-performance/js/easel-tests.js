
var phet = {};

$(document).ready( function() {
    // hook up activity changes so we can toggle sets of buttons
    _.each( [
        [ '#type-easel', '#type-custom' ], // Easel / Custom
        [ '#test-text', '#test-boxes' ] // Separate tests
    ], function( list ) {
        
        // handle toggling for each button
        _.each( list, function( selector ) {
            $( selector ).on( 'click', function( event, ui ) {
                _.each( list, function( button ) {
                    if( button == selector ) {
                        // select this button as active
                        $( button ).addClass( 'ui-btn-active' );
                    } else {
                        // and set others as inactive
                        $( button ).removeClass( 'ui-btn-active' );
                    }
                } );
            } );
        } );
    } );
    
    var main = $('#main');
    
    var fpsReadout = $('#fps-readout');
    var fpsCount = 10;
    var fpsIndex = 0;
    var timeEntries = _.map( _.range( fpsCount ), function( a ) { return 0; } );
    function updateFps() {
        var timestamp = Date.now();
        timeEntries[ fpsIndex ] = timestamp;
        fpsIndex = ( fpsIndex + 1 ) % fpsCount;
        var timeChange = ( timestamp - timeEntries[ fpsIndex ] ) / 1000;
        var fps = 10 / timeChange;
        fpsReadout.text( fps.toFixed( 1 ) );
    }
    
    function frame() {
        window.requestAnimationFrame( frame, main[0] );
        
        updateFps();
    }
    
    window.requestAnimationFrame( frame, main[0] );
    
    // for text testing: see http://www.whatwg.org/specs/web-apps/current-work/multipage/the-canvas-element.html#2dcontext
    // context: font, textAlign, textBaseline, direction
    // metrics: width, actualBoundingBoxLeft, actualBoundingBoxRight, etc.
} );
