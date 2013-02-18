// Copyright 2002-2012, University of Colorado

/**
 * DOM nodes. Currently lightweight handling
 *
 * @author Jonathan Olson <olsonsjc@gmail.com>
 */

var scenery = scenery || {};

(function(){
  "use strict";
  
  scenery.DOM = function( element, params ) {
    // unwrap from jQuery if that is passed in, for consistency
    if ( element && element.jquery ) {
      element = element[0];
    }
    
    this._element = element;
    this._$element = $( element );
    
    scenery.Node.call( this, params );
    
    this.invalidateDOM();
    
    this._$element.css( 'position', 'absolute' );
    this._$element.css( 'left', 0 );
    this._$element.css( 'top', 0 );
  };
  var DOM = scenery.DOM;
  
  DOM.prototype = phet.Object.create( scenery.Node.prototype );
  DOM.prototype.constructor = DOM;
  
  DOM.prototype.invalidatePaint = function( bounds ) {
    scenery.Node.prototype.invalidatePaint.call( this, bounds );
  };
  
  DOM.prototype.invalidateDOM = function() {
    // TODO: do we need to reset the CSS transform to get the proper bounds?
    
    // TODO: reset with the proper bounds here
    this.invalidateSelf( new phet.math.Bounds2( 0, 0, this._$element.width(), this._$element.height() ) );
  };
  
  DOM.prototype.addToDOMLayer = function( domLayer ) {
    // TODO: find better way to handle non-jquery and jquery-wrapped getters for the container. direct access for now ()
    domLayer.$div.append( this._element );
    
    // recompute the bounds
    this.invalidateDOM();
  };
  
  DOM.prototype.updateCSSTransform = function( transform ) {
    // something of the form matrix(...) as a String, with a Z translation to trigger hardware acceleration (hopefully)
    var cssTransform = transform.getMatrix().cssTransform();
    
    // notes on triggering hardware acceleration: http://creativejs.com/2011/12/day-2-gpu-accelerate-your-dom-elements/
    
    $( this._element ).css( {
      '-webkit-transform': cssTransform + ' translateZ(0)', // trigger hardware acceleration if possible
      '-moz-transform': cssTransform + ' translateZ(0)', // trigger hardware acceleration if possible
      '-ms-transform': cssTransform,
      '-o-transform': cssTransform,
      'transform': cssTransform,
      'transform-origin': 'top left', // at the origin of the component. consider 0px 0px instead. Critical, since otherwise this defaults to 50% 50%!!! see https://developer.mozilla.org/en-US/docs/CSS/transform-origin
      '-ms-transform-origin': 'top left' // TODO: do we need other platform-specific transform-origin styles?
    } );
  };
  
  DOM.prototype.hasSelf = function() {
    return true;
  };
  
  DOM.prototype.setElement = function( element ) {
    this._element = element;
    
    // TODO: bounds issue, since this will probably set to empty bounds and thus a repaint may not draw over it
    this.invalidateDOM();
    
    return this; // allow chaining
  };
  
  DOM.prototype.getElement = function() {
    return this._element;
  };
  
  DOM.prototype._mutatorKeys = [ 'element' ].concat( scenery.Node.prototype._mutatorKeys );
  
  DOM.prototype._supportedLayerTypes = [ scenery.LayerType.DOM ];
  
  Object.defineProperty( DOM.prototype, 'element', { set: DOM.prototype.setElement, get: DOM.prototype.getElement } );
  
})();


