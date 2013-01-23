// Copyright 2002-2012, University of Colorado

/**
 * Text
 *
 * @author Jonathan Olson
 */

var phet = phet || {};
phet.scene = phet.scene || {};
phet.scene.nodes = phet.scene.nodes || {};

(function(){
    phet.scene.nodes.Text = function( text ) {
        this._text = '';
        
        this.fontStyles = new Text.FontStyles(); // will be filled in later, due to dependency resolution
        
        phet.scene.Node.call( this );
        
        if( text !== undefined ) {
            this.setText( text );
        }
    };
    var Text = phet.scene.nodes.Text;
    
    Text.prototype = Object.create( phet.scene.Node.prototype );
    Text.prototype.constructor = Text;
    
    Text.prototype.setText = function( text ) {
        this._text = text;
        this.invalidateText();
    };
    
    Text.prototype.invalidateText = function() {
        // TODO: faster bounds determination? getBBox()?
        this.invalidateSelf( phet.scene.canvasTextBoundsAccurate( this._text, this.fontStyles ) );
    };
    
    Text.prototype.renderSelf = function( state ) {
        // TODO: add SVG / DOM support
        if( state.isCanvasState() ) {
            var layer = state.layer;
            var context = layer.context;
            if( this.fill ) {
                layer.setFillStyle( this.fill );
                layer.setFont( this.fontStyles.font );
                layer.setTextAlign( this.fontStyles.textAlign );
                layer.setTextBaseline( this.fontStyles.textBaseline );
                layer.setDirection( this.fontStyles.direction );
                
                context.fillText( this._text, 0, 0 );
            }
        }
    };
    
    Text.prototype.setFont = function( font ) {
        this.fontStyles.font = font;
        this.invalidateText();
    };
    
    Text.prototype.setTextAlign = function( textAlign ) {
        this.fontStyles.textAlign = textAlign;
        this.invalidateText();
    };
    
    Text.prototype.setTextBaseline = function( textBaseline ) {
        this.fontStyles.textBaseline = textBaseline;
        this.invalidateText();
    };
    
    Text.prototype.setDirection = function( direction ) {
        this.fontStyles.direction = direction;
        this.invalidateText();
    };
    
    Text.FontStyles = function( args ) {
        if( args === undefined ) {
            args = {};
        }
        this.font = args.font !== undefined ? args.font : '10px sans-serif',
        this.textAlign = args.textAlign !== undefined ? args.textAlign : 'start', // start, end, left, right, center
        this.textBaseline = args.textBaseline !== undefined ? args.textBaseline : 'alphabetic', // top, hanging, middle, alphabetic, ideographic, bottom
        this.direction = args.direction !== undefined ? args.direction : 'ltr' // ltr, rtl, inherit -- consider inherit deprecated, due to how we compute text bounds in an off-screen canvas
    }
})();


