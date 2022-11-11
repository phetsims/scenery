// Copyright 2022, University of Colorado Boulder

/**
 * A mixin that delays the mutation of a certain set of mutation keys until AFTER the super() call has fully finished.
 * This can be wrapped around a type where a mutate( { someKey: ... } ) would cause an error in the super(), and we
 * want to postpone that until after construction. e.g.:
 *
 * const SomeNode = DelayedMutate( 'SomeNode', [ 'someOption' ], class extends SuperNode {
 *   constructor( options ) {
 *     super( options );
 *
 *     this.someOptionProperty = new Property( something );
 *   }
 *
 *   set someOption( value: Something ) {
 *     this.someOptionProperty.value = value;
 *   }
 *
 *   get someOption(): Something {
 *     return this.someOptionProperty.value;
 *   }
 * } );
 *
 * If this was NOT done, the following would error out:
 *
 * new SomeNode( { someOption: something } )
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */

import { Node, NodeOptions, scenery } from '../imports.js';
import Constructor from '../../../phet-core/js/types/Constructor.js';
import { combineOptions } from '../../../phet-core/js/optionize.js';
import IntentionalAny from '../../../phet-core/js/types/IntentionalAny.js';

/**
 * @param name - A unique name for each call, which customizes the internal key names used to track state
 * @param keys - An array of the mutate option names that should be delayed
 * @param type - The class we're mixing into
 */
const DelayedMutate = <SuperType extends Constructor<Node>>( name: string, keys: string[], type: SuperType ) => { // eslint-disable-line @typescript-eslint/explicit-module-boundary-types
  // We typecast these to strings to satisfy the type-checker without large amounts of grief. It doesn't seem to be
  // able to parse that we're using the same keys for each call of this.
  const pendingOptionsKey = `_${name}PendingOptions` as '_fakePendingOptionsType';
  const isConstructedKey = `_${name}IsConstructed` as '_fakeIsConstructedType';

  return class DelayedMutateMixin extends type {

    // We need to store different fields in each class, so we use computed properties
    private [ isConstructedKey ]: boolean;
    private [ pendingOptionsKey ]: NodeOptions | undefined;

    public constructor( ...args: IntentionalAny[] ) {
      super( ...args );

      // Mark ourself as constructed, so further mutates will use all of the options
      this[ isConstructedKey ] = true;

      // Apply any options that we delayed
      this.mutate( this[ pendingOptionsKey ] );

      // Prevent memory leaks by tossing the options data that we've now used
      this[ pendingOptionsKey ] = undefined;
    }

    // Typescript doesn't want an override here, but we're overriding it
    public override mutate( options?: NodeOptions ): this {

      // If we're not constructed, we need to save the options for later
      // NOTE: If we haven't SET the constructed field yet, then it will be undefined (and falsy), so we do a check
      // for that here.
      if ( options && !this[ isConstructedKey ] ) {
        // Store delayed options. If we've provided the same option before, we'll want to use the most recent
        // (so a merge makes sense).
        this[ pendingOptionsKey ] = combineOptions<NodeOptions>( this[ pendingOptionsKey ] || {}, _.pick( options, keys ) );

        // We'll still want to mutate with the non-delayed options
        options = _.omit( options, keys );
      }

      return super.mutate( options );
    }
  };
};

scenery.register( 'DelayedMutate', DelayedMutate );
export default DelayedMutate;
