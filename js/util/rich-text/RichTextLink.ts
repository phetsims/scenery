// Copyright 2023, University of Colorado Boulder

/**
 * A link node in RichText - NOTE: This is NOT embedded for layout. Instead, link content will be added as children to this node,
 * and this will exist solely for the link functionality.
 *
 * @author Jonathan Olson <jonathan.olson@colorado.edu>
 */
import Pool, { TPoolable } from '../../../../phet-core/js/Pool.js';
import Tandem from '../../../../tandem/js/Tandem.js';
import { allowLinksProperty, FireListener, Node, openPopup, RichTextCleanable, RichTextHref, scenery, TInputListener, Voicing } from '../../imports.js';

export default class RichTextLink extends Voicing( RichTextCleanable( Node ) ) implements TPoolable {

  private fireListener: FireListener | null = null;
  private accessibleInputListener: TInputListener | null = null;
  private allowLinksListener: ( ( allowLinks: boolean ) => void ) | null = null;

  public constructor( innerContent: string, href: RichTextHref ) {
    super();

    // Voicing was already initialized in the super call, we do not want to initialize super again. But we do want to
    // initialize the RichText portion of the implementation.
    this.initialize( innerContent, href, false );

    // Mutate to make sure initialize doesn't clear this away
    this.mutate( {
      cursor: 'pointer',
      tagName: 'a'
    } );
  }

  /**
   * Set up this state. First construction does not need to use super.initialize() because the constructor has done
   * that for us. But repeated initialization with Poolable will need to initialize super again.
   */
  public override initialize( innerContent: string, href: RichTextHref, initializeSuper = true ): this {

    if ( initializeSuper ) {
      super.initialize();
    }

    // pdom - open the link in the new tab when activated with a keyboard.
    // also see https://github.com/phetsims/joist/issues/430
    this.innerContent = innerContent;

    this.voicingNameResponse = innerContent;

    // If our href is a function, it should be called when the user clicks on the link
    if ( typeof href === 'function' ) {
      this.fireListener = new FireListener( {
        fire: href,
        tandem: Tandem.OPT_OUT
      } );
      this.addInputListener( this.fireListener );
      this.setPDOMAttribute( 'href', '#' ); // Required so that the click listener will get called.
      this.setPDOMAttribute( 'target', '_self' ); // This is the default (easier than conditionally removing)
      this.accessibleInputListener = {
        click: event => {
          event.domEvent && event.domEvent.preventDefault();

          href();
        }
      };
      this.addInputListener( this.accessibleInputListener );
    }
    // Otherwise our href is a {string}, and we should open a window pointing to it (assuming it's a URL)
    else {
      this.fireListener = new FireListener( {
        fire: event => {
          if ( event.isFromPDOM() ) {

            // prevent default from pdom activation so we don't also open a new tab from native DOM input on a link
            event.domEvent!.preventDefault();
          }
          // @ts-expect-error TODO TODO TODO this is a bug! How do we handle this?
          self._linkEventsHandled && event.handle();
          openPopup( href ); // open in a new window/tab
        },
        tandem: Tandem.OPT_OUT
      } );
      this.addInputListener( this.fireListener );
      this.setPDOMAttribute( 'href', href );
      this.setPDOMAttribute( 'target', '_blank' );

      this.allowLinksListener = ( allowLinks: boolean ) => {
        this.visible = allowLinks;
      };
      allowLinksProperty.link( this.allowLinksListener );
    }

    return this;
  }

  /**
   * Cleans references that could cause memory leaks (as those things may contain other references).
   */
  public override clean(): void {
    super.clean();

    if ( this.fireListener ) {
      this.removeInputListener( this.fireListener );
      this.fireListener.dispose();
    }
    this.fireListener = null;
    if ( this.accessibleInputListener ) {
      this.removeInputListener( this.accessibleInputListener );
      this.accessibleInputListener = null;
    }
    if ( this.allowLinksListener ) {
      allowLinksProperty.unlink( this.allowLinksListener );
      this.allowLinksListener = null;
    }
  }

  public freeToPool(): void {
    RichTextLink.pool.freeToPool( this );
  }

  public static readonly pool = new Pool( RichTextLink );
}

scenery.register( 'RichTextLink', RichTextLink );
