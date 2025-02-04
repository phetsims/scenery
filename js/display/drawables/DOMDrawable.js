// Copyright 2016-2025, University of Colorado Boulder

/**
 * DOM renderer for DOM nodes.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import Poolable from '../../../../phet-core/js/Poolable.js';
import DOMSelfDrawable from '../../display/DOMSelfDrawable.js';
import scenery from '../../scenery.js';
import Utils from '../../util/Utils.js';

class DOMDrawable extends DOMSelfDrawable {
  /**
   * @param {number} renderer - Renderer bitmask, see Renderer's documentation for more details.
   * @param {Instance} instance
   */
  constructor( renderer, instance ) {
    super( renderer, instance );

    // Apply CSS needed for future CSS transforms to work properly.
    Utils.prepareForTransform( this.domElement );
  }

  /**
   * @public
   * @override
   *
   * @param {number} renderer
   * @param {Instance} instance
   */
  initialize( renderer, instance ) {
    super.initialize( renderer, instance );

    // @public {HTMLElement} - Our primary DOM element. This is exposed as part of the DOMSelfDrawable API.
    this.domElement = this.node._container;
  }

  /**
   * Updates our DOM element so that its appearance matches our node's representation.
   * @protected
   *
   * This implements part of the DOMSelfDrawable required API for subtypes.
   */
  updateDOM() {
    if ( this.transformDirty && !this.node._preventTransform ) {
      Utils.applyPreparedTransform( this.getTransformMatrix(), this.domElement );
    }

    // clear all of the dirty flags
    this.transformDirty = false;
  }

  /**
   * Disposes the drawable.
   * @public
   * @override
   */
  dispose() {
    super.dispose();

    this.domElement = null;
  }
}

scenery.register( 'DOMDrawable', DOMDrawable );

Poolable.mixInto( DOMDrawable );

export default DOMDrawable;