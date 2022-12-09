// Copyright 2018-2022, University of Colorado Boulder

/**
 * A property that is true when the node appears on the given display.
 *
 * Note that a node can appear on a display even after it has been removed from the scene graph, if
 * Display.updateDisplay has not yet been called since it was removed. So generally this Property will only update
 * as a result of Display.updateDisplay() being called.
 *
 * Be careful to dispose of these, since it WILL result in a permanent memory leak otherwise (Instance objects are
 * pooled, and if the listener is not removed, it will stay around forever).
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import BooleanProperty from '../../../axon/js/BooleanProperty.js';
import merge from '../../../phet-core/js/merge.js';
import { scenery } from '../imports.js';

class DisplayedProperty extends BooleanProperty {
  /**
   * @public
   * @extends {Property<boolean>}
   *
   * @param {scenery.Node} node
   * @param {Object} [options] - Passed through to the BooleanProperty
   */
  constructor( node, options ) {

    options = merge( {
      display: null // {Display|null} if null, this will check on any Display
    }, options );

    super( false, options );

    // @private {Node}
    this.node = node;

    // @private {Display|null}
    this.display = options.display;

    // @private {function}
    this.updateListener = this.updateValue.bind( this );
    this.changedInstanceListener = this.changedInstance.bind( this );

    node.changedInstanceEmitter.addListener( this.changedInstanceListener );
    // node.pdomDisplaysEmitter.addListener( this.updateListener ); // TODO support pdom visibility, https://github.com/phetsims/scenery/issues/1167

    // Add any instances the node may already have/
    const instances = node.instances;
    for ( let i = 0; i < instances.length; i++ ) {
      this.changedInstance( instances[ i ], true );
    }
  }

  /**
   * Checks whether the node was displayed and updates the value of this Property.
   * @private
   */
  updateValue() {
    this.value = this.node.wasVisuallyDisplayed( this.display );

    // TODO support pdom visibility, https://github.com/phetsims/scenery/issues/1167
    // this.value = this.node.wasVisuallyDisplayed( this.display ) || this.node.isPDOMDisplayed();
  }

  /**
   * Called when an instance is changed or added (based on the boolean flag).
   * @private
   *
   * @param {Instance} instance
   * @param {boolean} added
   */
  changedInstance( instance, added ) {
    if ( added ) {
      instance.visibleEmitter.addListener( this.updateListener );
    }
    else {
      instance.visibleEmitter.removeListener( this.updateListener );
    }

    this.updateValue();
  }

  /**
   * Releases references to avoid memory leaks.
   * @public
   * @override
   */
  dispose() {
    // Remove any instances the node may still have
    const instances = this.node.instances;
    for ( let i = 0; i < instances.length; i++ ) {
      this.changedInstance( instances[ i ], false );
    }

    this.node.changedInstanceEmitter.removeListener( this.changedInstanceListener );

    // TODO support pdom visibility, https://github.com/phetsims/scenery/issues/1167
    // this.node.pdomDisplaysEmitter.removeListener( this.updateListener );

    super.dispose();
  }
}

scenery.register( 'DisplayedProperty', DisplayedProperty );
export default DisplayedProperty;